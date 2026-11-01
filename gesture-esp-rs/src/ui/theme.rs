use macroquad::color::Color;
use crate::fsm::FsmState;

pub struct TacticalPalette;

impl TacticalPalette {
    pub const CRIMSON_LOCK: Color = Color::new(1.00, 0.165, 0.329, 1.0);
    pub const TACTICAL_CYAN: Color = Color::new(0.00, 0.941, 1.00, 1.0);
    pub const AMBER_DWELL: Color = Color::new(1.00, 0.722, 0.00, 1.0);
    pub const NEON_GREEN: Color = Color::new(0.00, 1.00, 0.40, 1.0);
    pub const MUTED_SLATE: Color = Color::new(0.424, 0.478, 0.537, 1.0);

    pub const ESP_BOX_TINT: Color = Color::new(0.00, 1.00, 0.502, 0.06);
    pub const HUD_PANEL_BG: Color = Color::new(0.031, 0.047, 0.071, 0.84);
    pub const HUD_BORDER_DIM: Color = Color::new(0.18, 0.26, 0.32, 0.75);
    pub const TEXT_PRIMARY: Color = Color::new(0.93, 0.97, 1.00, 1.0);
    pub const TEXT_DIM: Color = Color::new(0.62, 0.71, 0.78, 1.0);

    #[inline]
    pub fn color_for_state(state: &FsmState) -> Color {
        match state {
            FsmState::Locked { .. } => Self::CRIMSON_LOCK,
            FsmState::Armed => Self::TACTICAL_CYAN,
            FsmState::Dwelling { .. } => Self::AMBER_DWELL,
            FsmState::Continuous { .. } => Self::NEON_GREEN,
            FsmState::Cooldown { .. } => Self::MUTED_SLATE,
        }
    }
}
