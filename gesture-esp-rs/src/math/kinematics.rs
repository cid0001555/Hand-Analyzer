use glam::Vec3A;

use crate::math::normalizer::CanonicalHand;

pub const FINGER_JOINTS: [(usize, usize, usize, usize); 5] = [
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
];

pub const CANONICAL_PALM_METERS: f32 = 0.090;
pub const EXTENDED_PIP_MAX_DEG: f32 = 38.0;
pub const EXTENDED_DIP_MAX_DEG: f32 = 36.0;
pub const CURLED_PIP_MIN_DEG: f32 = 62.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum FingerId {
    Thumb = 0,
    Index = 1,
    Middle = 2,
    Ring = 3,
    Pinky = 4,
}

#[derive(Debug, Clone, Copy)]
pub struct FingerFlexion {
    pub pip_deg: f32,
    pub dip_deg: f32,
    pub is_extended: bool,
    pub is_curled: bool,
}

#[derive(Debug, Clone)]
pub struct HandKinematics {
    pub fingers: [FingerFlexion; 5],
    pub extended_mask: [bool; 5],
    pub curled_mask: [bool; 5],
    pub pinch_index_dist: f32,
    pub pinch_tip_only_dist: f32,
    pub pinch_aperture_deg: f32,
    pub pinch_middle_dist: f32,
    pub middle_aperture_deg: f32,
    pub index_middle_gap: f32,
    pub thumb_index_palm_ratio: f32,
    pub palm_normal: Vec3A,
    pub thumb_screen_vec: Vec3A,
}

impl HandKinematics {
    pub fn compute(joints: &[Vec3A; 21], canonical: &CanonicalHand) -> Self {
        let p = &canonical.scale_normalized;
        let mut fingers = [FingerFlexion {
            pip_deg: 0.0,
            dip_deg: 0.0,
            is_extended: false,
            is_curled: false,
        }; 5];
        let mut extended_mask = [false; 5];
        let mut curled_mask = [false; 5];

        for (idx, &(a, b, c, d)) in FINGER_JOINTS.iter().enumerate() {
            let v1 = joints[b] - joints[a];
            let v2 = joints[c] - joints[b];
            let v3 = joints[d] - joints[c];

            let pip_deg = safe_angle_deg(v1, v2);
            let dip_deg = safe_angle_deg(v2, v3);

            let (is_extended, is_curled) = if idx == FingerId::Thumb as usize {
                let thumb_palm_clearance = (p[4] - p[9]).length();
                let extended = pip_deg < EXTENDED_PIP_MAX_DEG
                    && dip_deg < EXTENDED_DIP_MAX_DEG
                    && thumb_palm_clearance > 0.35;
                let curled = pip_deg > 42.0 || dip_deg > 40.0 || thumb_palm_clearance < 0.30;
                (extended, curled)
            } else {
                let extended = pip_deg < EXTENDED_PIP_MAX_DEG && dip_deg < EXTENDED_DIP_MAX_DEG;
                let curled = pip_deg > CURLED_PIP_MIN_DEG;
                (extended, curled)
            };

            fingers[idx] = FingerFlexion {
                pip_deg,
                dip_deg,
                is_extended,
                is_curled,
            };
            extended_mask[idx] = is_extended;
            curled_mask[idx] = is_curled;
        }

        let delta_4_8 = damp_z(p[4] - p[8]);
        let thumb_index_palm_ratio = delta_4_8.length();
        let pinch_index_dist = thumb_index_palm_ratio * CANONICAL_PALM_METERS;
        let pinch_tip_only_dist = pinch_index_dist;

        let v_thumb_from_idx_mcp = damp_z(p[4] - p[5]);
        let v_index_from_idx_mcp = damp_z(p[8] - p[5]);
        let pinch_aperture_deg = chord_aperture_deg(v_thumb_from_idx_mcp, v_index_from_idx_mcp);

        let delta_4_12 = damp_z(p[4] - p[12]);
        let pinch_middle_dist = delta_4_12.length() * CANONICAL_PALM_METERS;
        let middle_aperture_deg = chord_aperture_deg(damp_z(p[4] - p[9]), damp_z(p[12] - p[9]));

        let index_middle_gap = (p[8] - p[12]).length();
        let thumb_screen_vec = p[4] - p[2];

        Self {
            fingers,
            extended_mask,
            curled_mask,
            pinch_index_dist,
            pinch_tip_only_dist,
            pinch_aperture_deg,
            pinch_middle_dist,
            middle_aperture_deg,
            index_middle_gap,
            thumb_index_palm_ratio,
            palm_normal: canonical.palm_normal,
            thumb_screen_vec,
        }
    }

    #[inline]
    pub fn palm_distance(canonical: &CanonicalHand, i: usize, j: usize) -> f32 {
        (canonical.scale_normalized[i] - canonical.scale_normalized[j]).length()
    }
}

#[inline]
fn damp_z(v: Vec3A) -> Vec3A {
    Vec3A::new(v.x, v.y, v.z * 0.35)
}

#[inline]
fn chord_aperture_deg(u: Vec3A, v: Vec3A) -> f32 {
    let denom = u.length() + v.length();
    if denom < 1.0e-6 {
        return 0.0;
    }
    let chord = (u - v).length();
    2.0 * chord.atan2(denom).to_degrees()
}

#[inline]
fn safe_angle_deg(u: Vec3A, v: Vec3A) -> f32 {
    let len_sq = u.length_squared() * v.length_squared();
    if len_sq < 1.0e-12 {
        return 0.0;
    }
    u.angle_between(v).to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::normalizer::HandNormalizer;

    #[test]
    fn test_pinch_kinematics() {
        let mut joints = [Vec3A::ZERO; 21];
        joints[0] = Vec3A::new(0.0, 0.0, 0.0);
        joints[5] = Vec3A::new(-0.03, 0.09, 0.0);
        joints[9] = Vec3A::new(0.0, 0.10, 0.0);
        joints[17] = Vec3A::new(0.05, 0.08, 0.0);

        joints[6] = Vec3A::new(-0.03, 0.13, 0.0);
        joints[7] = Vec3A::new(-0.03, 0.16, 0.0);
        joints[8] = Vec3A::new(-0.03, 0.19, 0.0);

        joints[10] = Vec3A::new(0.0, 0.13, 0.0);
        joints[11] = Vec3A::new(0.0, 0.12, -0.03);
        joints[12] = Vec3A::new(0.0, 0.09, -0.02);

        joints[13] = Vec3A::new(0.025, 0.09, 0.0);
        joints[14] = Vec3A::new(0.025, 0.12, 0.0);
        joints[15] = Vec3A::new(0.025, 0.11, -0.03);
        joints[16] = Vec3A::new(0.025, 0.08, -0.02);

        joints[18] = Vec3A::new(0.05, 0.11, 0.0);
        joints[19] = Vec3A::new(0.05, 0.10, -0.03);
        joints[20] = Vec3A::new(0.05, 0.07, -0.02);

        joints[1] = Vec3A::new(-0.02, 0.03, 0.0);
        joints[2] = Vec3A::new(-0.03, 0.06, -0.01);
        joints[3] = Vec3A::new(-0.02, 0.09, -0.015);
        joints[4] = Vec3A::new(-0.01, 0.115, -0.015);

        let canonical_tucked = HandNormalizer::normalize(&joints);
        let kin_tucked = HandKinematics::compute(&joints, &canonical_tucked);
        assert!(kin_tucked.pinch_aperture_deg > 35.0 && kin_tucked.pinch_index_dist > 0.060);

        joints[4] = Vec3A::new(-0.022, 0.135, -0.005);
        let canonical_parallel = HandNormalizer::normalize(&joints);
        let kin_parallel = HandKinematics::compute(&joints, &canonical_parallel);
        assert!(kin_parallel.pinch_aperture_deg > 35.0 && kin_parallel.pinch_index_dist > 0.045);

        joints[6] = Vec3A::new(-0.03, 0.13, -0.01);
        joints[7] = Vec3A::new(-0.03, 0.15, -0.025);
        joints[8] = Vec3A::new(-0.03, 0.145, -0.035);
        joints[3] = Vec3A::new(-0.03, 0.11, -0.025);
        joints[4] = Vec3A::new(-0.03, 0.140, -0.035);

        let canonical_pinch = HandNormalizer::normalize(&joints);
        let kin_pinch = HandKinematics::compute(&joints, &canonical_pinch);
        assert!(kin_pinch.pinch_aperture_deg < 12.0 && kin_pinch.pinch_index_dist < 0.020);
    }
}
