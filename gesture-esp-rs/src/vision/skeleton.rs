use std::f32::consts::FRAC_PI_2;
use glam::Vec3A;
use serde::{Deserialize, Serialize};

pub const HAND_CONNECTIONS: [(usize, usize); 21] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Handedness {
    Left,
    Right,
}

impl Handedness {
    #[inline]
    pub fn from_score(score: f32, mirrored: bool) -> Self {
        let raw_right = score >= 0.5;
        let is_right = if mirrored { !raw_right } else { raw_right };
        if is_right {
            Self::Right
        } else {
            Self::Left
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Left => "LEFT",
            Self::Right => "RIGHT",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OrientedRoi {
    pub center_x: f32,
    pub center_y: f32,
    pub size_px: f32,
    pub rotation_rad: f32,
}

impl OrientedRoi {
    pub fn from_landmarks(
        screen_landmarks: &[Vec3A; 21],
        frame_width: u32,
        frame_height: u32,
    ) -> Self {
        let fw = frame_width as f32;
        let fh = frame_height as f32;

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for lm in screen_landmarks {
            let px = lm.x * fw;
            let py = lm.y * fh;
            min_x = min_x.min(px);
            max_x = max_x.max(px);
            min_y = min_y.min(py);
            max_y = max_y.max(py);
        }

        let center_x = 0.5 * (min_x + max_x);
        let center_y = 0.5 * (min_y + max_y);
        let span = (max_x - min_x).max(max_y - min_y).max(32.0);
        let size_px = (span * 1.45).clamp(48.0, fw.max(fh) * 1.5);

        let wrist_px = (screen_landmarks[0].x * fw, screen_landmarks[0].y * fh);
        let mid_mcp_px = (screen_landmarks[9].x * fw, screen_landmarks[9].y * fh);
        let dx = mid_mcp_px.0 - wrist_px.0;
        let dy = mid_mcp_px.1 - wrist_px.1;
        let rotation_rad = normalize_radians(FRAC_PI_2 - (-dy).atan2(dx));

        Self {
            center_x,
            center_y,
            size_px,
            rotation_rad,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HandSkeleton {
    pub screen_landmarks: [Vec3A; 21],
    pub world_landmarks: [Vec3A; 21],
    pub presence_score: f32,
    pub handedness: Handedness,
    pub roi: OrientedRoi,
    pub used_palm_detector: bool,
}

#[inline]
pub fn normalize_radians(angle: f32) -> f32 {
    let two_pi = std::f32::consts::TAU;
    angle - two_pi * ((angle + std::f32::consts::PI) / two_pi).floor()
}
