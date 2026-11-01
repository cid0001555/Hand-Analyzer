use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::Instant;

use csv::{ReaderBuilder, WriterBuilder};
use tracing::info;

use crate::classifier::mlp::MlpClassifier;
use crate::error::DomainResult;
use crate::math::CANONICAL_FEATURE_DIM;

pub const DEFAULT_SAMPLES_PER_RECORDING: usize = 200;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecorderPhase {
    WaitingForTrigger { target_label: String },
    Capturing {
        target_label: String,
        captured: usize,
        total: usize,
    },
    Completed {
        target_label: String,
        accuracy: u32,
    },
}

pub struct PoseDatasetRecorder {
    target_label: String,
    target_samples: usize,
    buffer: Vec<[f32; CANONICAL_FEATURE_DIM]>,
    capturing: bool,
    completed_accuracy_permille: Option<u32>,
    csv_path: PathBuf,
    mlp_out_path: PathBuf,
}

impl PoseDatasetRecorder {
    pub fn new(
        gesture_name: &str,
        csv_path: impl AsRef<Path>,
        mlp_out_path: impl AsRef<Path>,
    ) -> Self {
        let normalized_label = gesture_name.trim().to_ascii_uppercase();
        Self {
            target_label: normalized_label,
            target_samples: DEFAULT_SAMPLES_PER_RECORDING,
            buffer: Vec::with_capacity(DEFAULT_SAMPLES_PER_RECORDING),
            capturing: false,
            completed_accuracy_permille: None,
            csv_path: csv_path.as_ref().to_path_buf(),
            mlp_out_path: mlp_out_path.as_ref().to_path_buf(),
        }
    }

    pub fn trigger_capture(&mut self) {
        if !self.capturing {
            self.buffer.clear();
            self.capturing = true;
            self.completed_accuracy_permille = None;
            info!(
                gesture = %self.target_label,
                samples = self.target_samples,
                "Recording started"
            );
        }
    }

    pub fn phase(&self) -> RecorderPhase {
        if self.capturing {
            RecorderPhase::Capturing {
                target_label: self.target_label.clone(),
                captured: self.buffer.len(),
                total: self.target_samples,
            }
        } else if let Some(acc) = self.completed_accuracy_permille {
            RecorderPhase::Completed {
                target_label: self.target_label.clone(),
                accuracy: acc,
            }
        } else {
            RecorderPhase::WaitingForTrigger {
                target_label: self.target_label.clone(),
            }
        }
    }

    pub fn push_canonical_sample(
        &mut self,
        features: &[f32; CANONICAL_FEATURE_DIM],
    ) -> DomainResult<Option<MlpClassifier>> {
        if !self.capturing {
            return Ok(None);
        }

        self.buffer.push(*features);
        if self.buffer.len() < self.target_samples {
            return Ok(None);
        }

        self.capturing = false;
        let started = Instant::now();
        self.append_buffer_to_csv()?;

        let trained = train_mlp_from_csv(&self.csv_path, &self.mlp_out_path)?;
        let elapsed_ms = started.elapsed().as_secs_f32() * 1000.0;
        self.completed_accuracy_permille = Some((trained.1 * 1000.0).round() as u32);

        info!(
            gesture = %self.target_label,
            accuracy_pct = trained.1 * 100.0,
            train_ms = elapsed_ms,
            model_path = %self.mlp_out_path.display(),
            "Saved dataset and trained MLP"
        );

        Ok(Some(trained.0))
    }

    fn append_buffer_to_csv(&self) -> DomainResult<()> {
        if let Some(parent) = self.csv_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let write_header = !self.csv_path.exists()
            || fs::metadata(&self.csv_path)
                .map(|m| m.len() == 0)
                .unwrap_or(true);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.csv_path)?;

        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);

        if write_header {
            let mut header: Vec<String> = (0..CANONICAL_FEATURE_DIM)
                .map(|i| format!("f{i}"))
                .collect();
            header.push("label".to_string());
            writer.write_record(&header)?;
        }

        for sample in &self.buffer {
            let mut row: Vec<String> = sample.iter().map(|v| format!("{v:.6}")).collect();
            row.push(self.target_label.clone());
            writer.write_record(&row)?;
        }
        writer.flush()?;
        Ok(())
    }
}

pub fn train_mlp_from_csv(
    csv_path: &Path,
    mlp_out_path: &Path,
) -> DomainResult<(MlpClassifier, f32)> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)?;

    let mut raw_rows: Vec<([f32; CANONICAL_FEATURE_DIM], String)> = Vec::new();
    let mut label_set: BTreeSet<String> = BTreeSet::new();

    for record_res in reader.records() {
        let record = record_res?;
        if record.len() < CANONICAL_FEATURE_DIM + 1 {
            continue;
        }
        let mut feats = [0.0f32; CANONICAL_FEATURE_DIM];
        let mut valid = true;
        for (i, slot) in feats.iter_mut().enumerate() {
            if let Ok(val) = record[i].parse::<f32>() {
                *slot = val;
            } else {
                valid = false;
                break;
            }
        }
        if !valid {
            continue;
        }
        let label = record[CANONICAL_FEATURE_DIM].trim().to_ascii_uppercase();
        label_set.insert(label.clone());
        raw_rows.push((feats, label));
    }

    if label_set.len() == 1 {
        let bg_label = "__BACKGROUND__".to_string();
        label_set.insert(bg_label.clone());
        for s in 0..DEFAULT_SAMPLES_PER_RECORDING {
            let mut bg = [0.0f32; CANONICAL_FEATURE_DIM];
            let phase = (s as f32) * 0.13;
            for (k, slot) in bg.iter_mut().enumerate() {
                *slot = ((k as f32 * 0.37 + phase).sin()) * 0.85;
            }
            raw_rows.push((bg, bg_label.clone()));
        }
    }

    let class_labels: Vec<String> = label_set.into_iter().collect();
    let indexed_samples: Vec<([f32; CANONICAL_FEATURE_DIM], usize)> = raw_rows
        .into_iter()
        .filter_map(|(feats, lbl)| {
            class_labels
                .iter()
                .position(|c| c == &lbl)
                .map(|idx| (feats, idx))
        })
        .collect();

    let (mlp, accuracy) = MlpClassifier::train(&indexed_samples, class_labels, 90, 32, 0.045)?;
    mlp.save_to_file(mlp_out_path)?;
    Ok((mlp, accuracy))
}
