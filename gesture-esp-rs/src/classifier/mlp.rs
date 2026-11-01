use std::fs;
use std::path::Path;

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::error::{DomainResult, GestureEspError};
use crate::math::CANONICAL_FEATURE_DIM;

const HIDDEN_1: usize = 128;
const HIDDEN_2: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MlpSerializedPayload {
    class_labels: Vec<String>,
    w1_shape: (usize, usize),
    w1_data: Vec<f32>,
    b1_data: Vec<f32>,
    w2_shape: (usize, usize),
    w2_data: Vec<f32>,
    b2_data: Vec<f32>,
    w3_shape: (usize, usize),
    w3_data: Vec<f32>,
    b3_data: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct MlpClassifier {
    class_labels: Vec<String>,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
    w3: Array2<f32>,
    b3: Array1<f32>,
}

impl MlpClassifier {
    pub fn new_initialized(class_labels: Vec<String>, seed: u64) -> DomainResult<Self> {
        if class_labels.len() < 2 {
            return Err(GestureEspError::Classifier(
                "at least 2 classes required".to_string(),
            ));
        }

        let num_classes = class_labels.len();
        let mut rng = XorShift64::new(seed);

        let w1 = he_init_matrix(CANONICAL_FEATURE_DIM, HIDDEN_1, &mut rng);
        let b1 = Array1::zeros(HIDDEN_1);
        let w2 = he_init_matrix(HIDDEN_1, HIDDEN_2, &mut rng);
        let b2 = Array1::zeros(HIDDEN_2);
        let w3 = he_init_matrix(HIDDEN_2, num_classes, &mut rng);
        let b3 = Array1::zeros(num_classes);

        Ok(Self {
            class_labels,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        })
    }

    #[inline]
    pub fn class_labels(&self) -> &[String] {
        &self.class_labels
    }

    pub fn predict(&self, features: &[f32; CANONICAL_FEATURE_DIM]) -> (&str, f32) {
        let probs = self.forward_probabilities(features);
        let mut best_idx = 0usize;
        let mut best_prob = f32::NEG_INFINITY;
        for (i, &p) in probs.iter().enumerate() {
            if p > best_prob {
                best_prob = p;
                best_idx = i;
            }
        }
        (&self.class_labels[best_idx], best_prob.clamp(0.0, 1.0))
    }

    pub fn forward_probabilities(&self, features: &[f32; CANONICAL_FEATURE_DIM]) -> Array1<f32> {
        let x = Array1::from_vec(features.to_vec());
        let mut h1 = x.dot(&self.w1) + &self.b1;
        h1.mapv_inplace(|v| v.max(0.0));

        let mut h2 = h1.dot(&self.w2) + &self.b2;
        h2.mapv_inplace(|v| v.max(0.0));

        let logits = h2.dot(&self.w3) + &self.b3;
        softmax_1d(&logits)
    }

    pub fn train(
        samples: &[([f32; CANONICAL_FEATURE_DIM], usize)],
        class_labels: Vec<String>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
    ) -> DomainResult<(Self, f32)> {
        if samples.is_empty() {
            return Err(GestureEspError::Classifier("empty dataset".to_string()));
        }

        let mut model = Self::new_initialized(class_labels, 0x9E37_79B9_7F4A_7C15)?;
        let num_classes = model.class_labels.len();
        let bsz = batch_size.max(1).min(samples.len());
        let momentum = 0.90f32;
        let weight_decay = 1.0e-4f32;

        let mut vw1 = Array2::<f32>::zeros(model.w1.raw_dim());
        let mut vb1 = Array1::<f32>::zeros(model.b1.raw_dim());
        let mut vw2 = Array2::<f32>::zeros(model.w2.raw_dim());
        let mut vb2 = Array1::<f32>::zeros(model.b2.raw_dim());
        let mut vw3 = Array2::<f32>::zeros(model.w3.raw_dim());
        let mut vb3 = Array1::<f32>::zeros(model.b3.raw_dim());

        let mut indices: Vec<usize> = (0..samples.len()).collect();
        let mut rng = XorShift64::new(0xD1B5_4A32_D192_ED03);

        for epoch in 0..epochs {
            let progress = (epoch as f32) / (epochs.max(1) as f32);
            let lr = learning_rate * (0.15 + 0.85 * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()));

            rng.shuffle(&mut indices);

            for chunk in indices.chunks(bsz) {
                let m = chunk.len();
                let inv_m = 1.0 / (m as f32);

                let mut x_batch = Array2::<f32>::zeros((m, CANONICAL_FEATURE_DIM));
                let mut y_batch = Vec::with_capacity(m);

                for (row_idx, &sample_idx) in chunk.iter().enumerate() {
                    let (ref feats, cls) = samples[sample_idx];
                    for col in 0..CANONICAL_FEATURE_DIM {
                        x_batch[[row_idx, col]] = feats[col];
                    }
                    y_batch.push(cls);
                }

                let z1 = x_batch.dot(&model.w1) + &model.b1;
                let a1 = z1.mapv(|v| v.max(0.0));

                let z2 = a1.dot(&model.w2) + &model.b2;
                let a2 = z2.mapv(|v| v.max(0.0));

                let z3 = a2.dot(&model.w3) + &model.b3;

                let mut dz3 = Array2::<f32>::zeros((m, num_classes));
                for r in 0..m {
                    let mut max_logit = f32::NEG_INFINITY;
                    for c in 0..num_classes {
                        max_logit = max_logit.max(z3[[r, c]]);
                    }
                    let mut sum_exp = 0.0f32;
                    for c in 0..num_classes {
                        let e = (z3[[r, c]] - max_logit).exp();
                        dz3[[r, c]] = e;
                        sum_exp += e;
                    }
                    let inv_sum = 1.0 / sum_exp.max(1.0e-12);
                    for c in 0..num_classes {
                        dz3[[r, c]] *= inv_sum;
                    }
                    dz3[[r, y_batch[r]]] -= 1.0;
                    for c in 0..num_classes {
                        dz3[[r, c]] *= inv_m;
                    }
                }

                let dw3 = a2.t().dot(&dz3) + (&model.w3 * weight_decay);
                let db3 = dz3.sum_axis(Axis(0));

                let da2 = dz3.dot(&model.w3.t());
                let mut dz2 = da2;
                dz2.zip_mut_with(&z2, |g, &z| {
                    if z <= 0.0 {
                        *g = 0.0;
                    }
                });
                let dw2 = a1.t().dot(&dz2) + (&model.w2 * weight_decay);
                let db2 = dz2.sum_axis(Axis(0));

                let da1 = dz2.dot(&model.w2.t());
                let mut dz1 = da1;
                dz1.zip_mut_with(&z1, |g, &z| {
                    if z <= 0.0 {
                        *g = 0.0;
                    }
                });
                let dw1 = x_batch.t().dot(&dz1) + (&model.w1 * weight_decay);
                let db1 = dz1.sum_axis(Axis(0));

                vw3 = &vw3 * momentum - &dw3 * lr;
                vb3 = &vb3 * momentum - &db3 * lr;
                model.w3 += &vw3;
                model.b3 += &vb3;

                vw2 = &vw2 * momentum - &dw2 * lr;
                vb2 = &vb2 * momentum - &db2 * lr;
                model.w2 += &vw2;
                model.b2 += &vb2;

                vw1 = &vw1 * momentum - &dw1 * lr;
                vb1 = &vb1 * momentum - &db1 * lr;
                model.w1 += &vw1;
                model.b1 += &vb1;
            }
        }

        let mut correct = 0usize;
        for (feats, target_cls) in samples {
            let (pred_label, _) = model.predict(feats);
            if pred_label == model.class_labels[*target_cls] {
                correct += 1;
            }
        }
        let accuracy = (correct as f32) / (samples.len() as f32);

        Ok((model, accuracy))
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>) -> DomainResult<()> {
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            fs::create_dir_all(parent)?;
        }
        let payload = MlpSerializedPayload {
            class_labels: self.class_labels.clone(),
            w1_shape: self.w1.dim(),
            w1_data: self.w1.iter().copied().collect(),
            b1_data: self.b1.to_vec(),
            w2_shape: self.w2.dim(),
            w2_data: self.w2.iter().copied().collect(),
            b2_data: self.b2.to_vec(),
            w3_shape: self.w3.dim(),
            w3_data: self.w3.iter().copied().collect(),
            b3_data: self.b3.to_vec(),
        };
        let encoded = bincode::serialize(&payload)?;
        fs::write(path_ref, encoded)?;
        Ok(())
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> DomainResult<Self> {
        let raw = fs::read(path.as_ref())?;
        let payload: MlpSerializedPayload = bincode::deserialize(&raw)?;

        let w1 = Array2::from_shape_vec(payload.w1_shape, payload.w1_data)
            .map_err(|e| GestureEspError::Classifier(e.to_string()))?;
        let b1 = Array1::from_vec(payload.b1_data);

        let w2 = Array2::from_shape_vec(payload.w2_shape, payload.w2_data)
            .map_err(|e| GestureEspError::Classifier(e.to_string()))?;
        let b2 = Array1::from_vec(payload.b2_data);

        let w3 = Array2::from_shape_vec(payload.w3_shape, payload.w3_data)
            .map_err(|e| GestureEspError::Classifier(e.to_string()))?;
        let b3 = Array1::from_vec(payload.b3_data);

        Ok(Self {
            class_labels: payload.class_labels,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        })
    }
}

#[inline]
fn softmax_1d(logits: &Array1<f32>) -> Array1<f32> {
    let max_val = logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let exps = logits.mapv(|x| (x - max_val).exp());
    let sum = exps.sum().max(1.0e-12);
    exps / sum
}

fn he_init_matrix(fan_in: usize, fan_out: usize, rng: &mut XorShift64) -> Array2<f32> {
    let limit = (6.0f32 / (fan_in as f32)).sqrt();
    Array2::from_shape_fn((fan_in, fan_out), |_| rng.next_f32_range(-limit, limit))
}

#[derive(Debug, Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xCAFE_BABE_DEAD_BEEF } else { seed },
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    fn next_f32_unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    #[inline]
    fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32_unit()
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_training() {
        let mut samples = Vec::new();
        for s in 0..80 {
            let jitter = (s as f32) * 0.001;
            let mut feat_a = [0.0f32; CANONICAL_FEATURE_DIM];
            let mut feat_b = [0.0f32; CANONICAL_FEATURE_DIM];
            for k in 0..CANONICAL_FEATURE_DIM {
                feat_a[k] = ((k as f32) * 0.05).sin() + jitter;
                feat_b[k] = ((k as f32) * 0.05).cos() - jitter;
            }
            samples.push((feat_a, 0usize));
            samples.push((feat_b, 1usize));
        }

        let labels = vec!["POSE_A".to_string(), "POSE_B".to_string()];
        let (mlp, acc) = MlpClassifier::train(&samples, labels, 60, 16, 0.05).unwrap();
        assert!(acc > 0.98);

        let (pred, conf) = mlp.predict(&samples[0].0);
        assert_eq!(pred, "POSE_A");
        assert!(conf > 0.90);
    }
}
