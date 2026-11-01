pub mod heuristic;
pub mod mlp;
pub mod recorder;

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::math::{CanonicalHand, HandKinematics};

pub use heuristic::HeuristicClassifier;
pub use mlp::MlpClassifier;
pub use recorder::{PoseDatasetRecorder, RecorderPhase};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GestureTag {
    Neutral,
    OpenPalm,
    Fist,
    Victory,
    TwoFingerTogether,
    Pointing,
    Shaka,
    RockOn,
    ThumbUp,
    ThumbDown,
    OkSignSlider,
    Custom(String),
}

impl GestureTag {
    pub fn as_key(&self) -> &str {
        match self {
            Self::Neutral => "NEUTRAL",
            Self::OpenPalm => "OPEN_PALM",
            Self::Fist => "FIST",
            Self::Victory => "VICTORY",
            Self::TwoFingerTogether => "TWO_FINGER_TOGETHER",
            Self::Pointing => "POINTING",
            Self::Shaka => "SHAKA",
            Self::RockOn => "ROCK_ON",
            Self::ThumbUp => "THUMB_UP",
            Self::ThumbDown => "THUMB_DOWN",
            Self::OkSignSlider => "OK_SIGN_SLIDER",
            Self::Custom(name) => name.as_str(),
        }
    }

    pub fn from_name(raw: &str) -> Self {
        match raw.trim().to_ascii_uppercase().as_str() {
            "NEUTRAL" | "NONE" | "IDLE" => Self::Neutral,
            "OPEN_PALM" => Self::OpenPalm,
            "FIST" => Self::Fist,
            "VICTORY" => Self::Victory,
            "TWO_FINGER_TOGETHER" => Self::TwoFingerTogether,
            "POINTING" => Self::Pointing,
            "SHAKA" => Self::Shaka,
            "ROCK_ON" => Self::RockOn,
            "THUMB_UP" => Self::ThumbUp,
            "THUMB_DOWN" => Self::ThumbDown,
            "OK_SIGN_SLIDER" => Self::OkSignSlider,
            other => Self::Custom(other.to_string()),
        }
    }

    #[inline]
    pub fn is_neutral(&self) -> bool {
        matches!(self, Self::Neutral)
    }
}

impl fmt::Display for GestureTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_key())
    }
}

#[derive(Debug, Clone)]
pub struct ClassificationOutput {
    pub tag: GestureTag,
    pub confidence: f32,
    pub from_mlp: bool,
}

pub struct HybridClassifier {
    heuristic: HeuristicClassifier,
    mlp: Option<MlpClassifier>,
    mlp_confidence_threshold: f32,
}

impl HybridClassifier {
    pub fn new(mlp_weights_path: Option<&Path>) -> Self {
        let mlp = mlp_weights_path.and_then(|path| {
            if !path.exists() {
                return None;
            }
            match MlpClassifier::load_from_file(path) {
                Ok(model) => {
                    info!(
                        classes = ?model.class_labels(),
                        path = %path.display(),
                        "Loaded MLP weights"
                    );
                    Some(model)
                }
                Err(err) => {
                    warn!(error = %err, path = %path.display(), "Failed to load MLP weights");
                    None
                }
            }
        });

        Self {
            heuristic: HeuristicClassifier,
            mlp,
            mlp_confidence_threshold: 0.88,
        }
    }

    pub fn set_mlp(&mut self, model: MlpClassifier) {
        self.mlp = Some(model);
    }

    pub fn classify(
        &self,
        canonical: &CanonicalHand,
        kinematics: &HandKinematics,
    ) -> ClassificationOutput {
        let (heur_tag, heur_conf) = self.heuristic.evaluate(canonical, kinematics);

        if let Some(mlp) = &self.mlp {
            let (label, prob) = mlp.predict(&canonical.feature_vector);
            if label != "__BACKGROUND__" && prob >= self.mlp_confidence_threshold {
                let mlp_tag = GestureTag::from_name(label);
                if matches!(mlp_tag, GestureTag::Custom(_)) || heur_tag.is_neutral() || prob > heur_conf {
                    return ClassificationOutput {
                        tag: mlp_tag,
                        confidence: prob,
                        from_mlp: true,
                    };
                }
            }
        }

        ClassificationOutput {
            tag: heur_tag,
            confidence: heur_conf,
            from_mlp: false,
        }
    }
}
