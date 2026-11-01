use glam::{Mat3A, Vec3A};

pub const CANONICAL_FEATURE_DIM: usize = 63;
const SCALE_EPSILON: f32 = 1.0e-6;

#[derive(Debug, Clone)]
pub struct CanonicalHand {
    pub scale_normalized: [Vec3A; 21],
    pub canonical_points: [Vec3A; 21],
    pub feature_vector: [f32; CANONICAL_FEATURE_DIM],
    pub palm_scale: f32,
    pub basis_x: Vec3A,
    pub basis_y: Vec3A,
    pub palm_normal: Vec3A,
}

pub struct HandNormalizer;

impl HandNormalizer {
    pub fn normalize(joints: &[Vec3A; 21]) -> CanonicalHand {
        let wrist = joints[0];

        let palm_vec = joints[9] - wrist;
        let palm_scale = palm_vec.length() + SCALE_EPSILON;
        let inv_scale = 1.0 / palm_scale;

        let mut scale_normalized = [Vec3A::ZERO; 21];
        for i in 0..21 {
            scale_normalized[i] = (joints[i] - wrist) * inv_scale;
        }

        let basis_y = scale_normalized[9]
            .try_normalize()
            .unwrap_or(Vec3A::Y);

        let knuckle_u = scale_normalized[5] - scale_normalized[17];

        let cross_uy = knuckle_u.cross(basis_y);
        let palm_normal = cross_uy.try_normalize().unwrap_or_else(|| {
            let fallback_ref = if basis_y.x.abs() < 0.9 {
                Vec3A::X
            } else {
                Vec3A::Z
            };
            fallback_ref.cross(basis_y).normalize_or_zero()
        });

        let basis_x = basis_y
            .cross(palm_normal)
            .try_normalize()
            .unwrap_or(Vec3A::X);

        let rot_transpose = Mat3A::from_cols(basis_x, basis_y, palm_normal).transpose();

        let mut canonical_points = [Vec3A::ZERO; 21];
        let mut feature_vector = [0.0f32; CANONICAL_FEATURE_DIM];

        for i in 0..21 {
            let q = rot_transpose * scale_normalized[i];
            canonical_points[i] = q;
            let base = i * 3;
            feature_vector[base] = q.x;
            feature_vector[base + 1] = q.y;
            feature_vector[base + 2] = q.z;
        }

        CanonicalHand {
            scale_normalized,
            canonical_points,
            feature_vector,
            palm_scale,
            basis_x,
            basis_y,
            palm_normal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn sample_hand() -> [Vec3A; 21] {
        [
            Vec3A::new(0.0, 0.0, 0.0),
            Vec3A::new(-0.03, 0.02, 0.0),
            Vec3A::new(-0.05, 0.05, 0.0),
            Vec3A::new(-0.06, 0.08, 0.0),
            Vec3A::new(-0.07, 0.11, 0.0),
            Vec3A::new(-0.03, 0.09, 0.0),
            Vec3A::new(-0.03, 0.13, 0.0),
            Vec3A::new(-0.03, 0.16, 0.0),
            Vec3A::new(-0.03, 0.19, 0.0),
            Vec3A::new(0.0, 0.095, 0.0),
            Vec3A::new(0.0, 0.14, 0.0),
            Vec3A::new(0.0, 0.175, 0.0),
            Vec3A::new(0.0, 0.205, 0.0),
            Vec3A::new(0.025, 0.09, 0.0),
            Vec3A::new(0.025, 0.13, 0.0),
            Vec3A::new(0.025, 0.16, 0.0),
            Vec3A::new(0.025, 0.19, 0.0),
            Vec3A::new(0.05, 0.08, 0.0),
            Vec3A::new(0.05, 0.115, 0.0),
            Vec3A::new(0.05, 0.14, 0.0),
            Vec3A::new(0.05, 0.165, 0.0),
        ]
    }

    #[test]
    fn test_similarity_invariance() {
        let base = sample_hand();
        let norm_a = HandNormalizer::normalize(&base);

        let rot = Quat::from_euler(glam::EulerRot::XYZ, 0.73, -1.12, 0.48);
        let translation = Vec3A::new(14.2, -8.7, 3.1);
        let scale = 3.7f32;

        let mut transformed = [Vec3A::ZERO; 21];
        for i in 0..21 {
            transformed[i] = Vec3A::from(rot * (glam::Vec3::from(base[i]) * scale)) + translation;
        }

        let norm_b = HandNormalizer::normalize(&transformed);

        for k in 0..CANONICAL_FEATURE_DIM {
            let diff = (norm_a.feature_vector[k] - norm_b.feature_vector[k]).abs();
            assert!(diff < 1.0e-4);
        }
    }
}
