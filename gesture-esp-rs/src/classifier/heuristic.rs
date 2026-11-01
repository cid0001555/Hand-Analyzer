use crate::classifier::GestureTag;
use crate::math::{CanonicalHand, FingerId, HandKinematics};

#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicClassifier;

impl HeuristicClassifier {
    pub fn evaluate(
        &self,
        canonical: &CanonicalHand,
        kin: &HandKinematics,
    ) -> (GestureTag, f32) {
        let [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext] = kin.extended_mask;
        let [_thumb_curl, index_curl, middle_curl, ring_curl, pinky_curl] = kin.curled_mask;

        let d_8_12 = HandKinematics::palm_distance(canonical, 8, 12);

        if middle_ext
            && ring_ext
            && pinky_ext
            && (kin.pinch_tip_only_dist < 0.048 || kin.thumb_index_palm_ratio < 0.50)
        {
            let tightness = (1.0 - (kin.pinch_tip_only_dist / 0.062)).clamp(0.0, 1.0);
            let conf = (0.88 + 0.11 * tightness).clamp(0.88, 0.99);
            return (GestureTag::OkSignSlider, conf);
        }

        if [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext] == [true, true, true, true, true]
        {
            let straightness = mean_extension_margin(kin, &[0, 1, 2, 3, 4]);
            return (GestureTag::OpenPalm, 0.90 + 0.095 * straightness);
        }

        if thumb_ext
            && !index_ext
            && !middle_ext
            && !ring_ext
            && pinky_ext
            && middle_curl
            && ring_curl
        {
            let straightness = mean_extension_margin(kin, &[0, 4]);
            return (GestureTag::Shaka, 0.90 + 0.09 * straightness);
        }

        if !thumb_ext
            && index_ext
            && !middle_ext
            && !ring_ext
            && pinky_ext
            && middle_curl
            && ring_curl
        {
            let straightness = mean_extension_margin(kin, &[1, 4]);
            return (GestureTag::RockOn, 0.89 + 0.10 * straightness);
        }

        if index_ext && middle_ext && !ring_ext && !pinky_ext && ring_curl && pinky_curl {
            let straightness = mean_extension_margin(kin, &[1, 2]);
            if d_8_12 > 0.24 {
                let spread_bonus = ((d_8_12 - 0.24) / 0.25).clamp(0.0, 1.0);
                return (
                    GestureTag::Victory,
                    (0.88 + 0.06 * straightness + 0.055 * spread_bonus).min(0.998),
                );
            } else if d_8_12 <= 0.21 {
                let touch_bonus = ((0.21 - d_8_12) / 0.12).clamp(0.0, 1.0);
                return (
                    GestureTag::TwoFingerTogether,
                    (0.88 + 0.06 * straightness + 0.055 * touch_bonus).min(0.998),
                );
            }
        }

        let index_pinch_flexing = kin.fingers[FingerId::Index as usize].pip_deg < 120.0
            && canonical.scale_normalized[8].length() > 0.55
            && kin.pinch_aperture_deg < 37.0
            && kin.pinch_index_dist < 0.060;
        if (index_ext || index_pinch_flexing)
            && !middle_ext
            && !ring_ext
            && !pinky_ext
            && middle_curl
            && ring_curl
            && pinky_curl
        {
            let straightness = if index_ext {
                mean_extension_margin(kin, &[FingerId::Index as usize])
            } else {
                0.75
            };
            return (GestureTag::Pointing, 0.89 + 0.10 * straightness);
        }

        if thumb_ext
            && !index_ext
            && !middle_ext
            && !ring_ext
            && !pinky_ext
            && index_curl
            && middle_curl
            && ring_curl
            && pinky_curl
        {
            let dy = kin.thumb_screen_vec.y;
            let vert_ratio = dy.abs() / (kin.thumb_screen_vec.length().max(1.0e-5));
            if dy < -0.12 && vert_ratio > 0.45 {
                let conf = (0.87 + 0.12 * vert_ratio).min(0.995);
                return (GestureTag::ThumbUp, conf);
            } else if dy > 0.12 && vert_ratio > 0.45 {
                let conf = (0.87 + 0.12 * vert_ratio).min(0.995);
                return (GestureTag::ThumbDown, conf);
            }
        }

        if !thumb_ext
            && !index_ext
            && !middle_ext
            && !ring_ext
            && !pinky_ext
            && index_curl
            && middle_curl
            && ring_curl
            && pinky_curl
        {
            let curl_strength = mean_curl_margin(kin, &[1, 2, 3, 4]);
            return (GestureTag::Fist, 0.89 + 0.10 * curl_strength);
        }

        (GestureTag::Neutral, 0.50)
    }
}

#[inline]
fn mean_extension_margin(kin: &HandKinematics, indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.5;
    }
    let mut sum = 0.0f32;
    for &i in indices {
        let pip_score = ((38.0 - kin.fingers[i].pip_deg) / 38.0).clamp(0.0, 1.0);
        let dip_score = ((36.0 - kin.fingers[i].dip_deg) / 36.0).clamp(0.0, 1.0);
        sum += 0.5 * (pip_score + dip_score);
    }
    (sum / (indices.len() as f32)).clamp(0.0, 1.0)
}

#[inline]
fn mean_curl_margin(kin: &HandKinematics, indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.5;
    }
    let mut sum = 0.0f32;
    for &i in indices {
        let score = ((kin.fingers[i].pip_deg - 62.0) / 55.0).clamp(0.0, 1.0);
        sum += score;
    }
    (sum / (indices.len() as f32)).clamp(0.0, 1.0)
}
