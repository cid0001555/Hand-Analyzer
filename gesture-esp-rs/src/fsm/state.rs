use std::time::Instant;
use glam::Vec3A;
use serde::{Deserialize, Serialize};

use crate::classifier::GestureTag;
use crate::config::{ActionSpec, ContinuousControlType, MediaCommandConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContinuousMode {
    MousePointer,
    ScrollWheel,
    VolumeSlider,
}

impl From<ContinuousControlType> for ContinuousMode {
    fn from(value: ContinuousControlType) -> Self {
        match value {
            ContinuousControlType::MousePointer => Self::MousePointer,
            ContinuousControlType::ScrollWheel => Self::ScrollWheel,
            ContinuousControlType::VolumeSlider => Self::VolumeSlider,
        }
    }
}

impl ContinuousMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::MousePointer => "AIR_MOUSE",
            Self::ScrollWheel => "2F_SCROLL",
            Self::VolumeSlider => "VOL_SLIDER",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FsmState {
    Locked {
        unlock_progress: f32,
        unlock_started: Option<Instant>,
    },
    Armed,
    Dwelling {
        candidate: GestureTag,
        started_at: Instant,
        progress: f32,
    },
    Continuous {
        mode: ContinuousMode,
        initial_anchor: Vec3A,
        initial_value: f32,
        pinch_active: bool,
    },
    Cooldown {
        last_gesture: GestureTag,
        until: Instant,
        require_neutral_reset: bool,
    },
}

impl FsmState {
    pub fn short_label(&self) -> &'static str {
        match self {
            Self::Locked { .. } => "LOCKED",
            Self::Armed => "ARMED",
            Self::Dwelling { .. } => "DWELLING",
            Self::Continuous { .. } => "CONTINUOUS",
            Self::Cooldown { .. } => "COOLDOWN",
        }
    }

    pub fn dwell_progress(&self) -> f32 {
        match self {
            Self::Locked {
                unlock_progress, ..
            } => *unlock_progress,
            Self::Dwelling { progress, .. } => *progress,
            _ => 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MouseBtn {
    Left,
    Right,
    Middle,
}

impl MouseBtn {
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "right" | "r" | "btn_right" => Self::Right,
            "middle" | "mid" | "m" | "btn_middle" => Self::Middle,
            _ => Self::Left,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaCmd {
    PlayPause,
    Mute,
    NextTrack,
    PrevTrack,
    VolumeUp,
    VolumeDown,
}

impl From<MediaCommandConfig> for MediaCmd {
    fn from(value: MediaCommandConfig) -> Self {
        match value {
            MediaCommandConfig::PlayPause => Self::PlayPause,
            MediaCommandConfig::Mute => Self::Mute,
            MediaCommandConfig::NextTrack => Self::NextTrack,
            MediaCommandConfig::PrevTrack => Self::PrevTrack,
            MediaCommandConfig::VolumeUp => Self::VolumeUp,
            MediaCommandConfig::VolumeDown => Self::VolumeDown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VirtualKey {
    Super,
    Ctrl,
    Shift,
    Alt,
    Tab,
    Enter,
    Escape,
    Space,
    Backspace,
    Left,
    Right,
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
    F(u8),
    Alpha(char),
    Digit(u8),
}

impl VirtualKey {
    pub fn parse(token: &str) -> Option<Self> {
        let t = token.trim().to_ascii_uppercase();
        match t.as_str() {
            "SUPER" | "WIN" | "LWIN" | "META" | "CMD" => Some(Self::Super),
            "CTRL" | "CONTROL" | "LCTRL" => Some(Self::Ctrl),
            "SHIFT" | "LSHIFT" => Some(Self::Shift),
            "ALT" | "LALT" => Some(Self::Alt),
            "TAB" => Some(Self::Tab),
            "ENTER" | "RETURN" => Some(Self::Enter),
            "ESC" | "ESCAPE" => Some(Self::Escape),
            "SPACE" => Some(Self::Space),
            "BACKSPACE" | "BKSP" => Some(Self::Backspace),
            "LEFT" | "ARROWLEFT" => Some(Self::Left),
            "RIGHT" | "ARROWRIGHT" => Some(Self::Right),
            "UP" | "ARROWUP" => Some(Self::Up),
            "DOWN" | "ARROWDOWN" => Some(Self::Down),
            "PGUP" | "PAGEUP" => Some(Self::PageUp),
            "PGDN" | "PAGEDOWN" => Some(Self::PageDown),
            "HOME" => Some(Self::Home),
            "END" => Some(Self::End),
            _ => {
                if let Some(rest) = t.strip_prefix('F') {
                    if let Ok(num) = rest.parse::<u8>() {
                        if (1..=12).contains(&num) {
                            return Some(Self::F(num));
                        }
                    }
                }
                if t.len() == 1 {
                    let ch = t.chars().next()?;
                    if ch.is_ascii_uppercase() {
                        return Some(Self::Alpha(ch));
                    }
                    if ch.is_ascii_digit() {
                        return Some(Self::Digit((ch as u8) - b'0'));
                    }
                }
                None
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum OsActionEvent {
    MovePointerRel { dx: i32, dy: i32 },
    MovePointerAbs { norm_x: f32, norm_y: f32 },
    MouseButton { button: MouseBtn, pressed: bool },
    MouseScroll { wheel_y: i32, wheel_x: i32 },
    Hotkey { keys: Vec<VirtualKey> },
    Media(MediaCmd),
    SetSystemVolume(f32),
    ExecuteShell(String),
}

impl OsActionEvent {
    pub fn from_action_spec(spec: &ActionSpec) -> Vec<Self> {
        match spec {
            ActionSpec::Media { command } => vec![Self::Media((*command).into())],
            ActionSpec::Hotkey { keys } => {
                let parsed: Vec<VirtualKey> = keys
                    .iter()
                    .filter_map(|k| VirtualKey::parse(k))
                    .collect();
                if parsed.is_empty() {
                    Vec::new()
                } else {
                    vec![Self::Hotkey { keys: parsed }]
                }
            }
            ActionSpec::Shell { command } => vec![Self::ExecuteShell(command.clone())],
            ActionSpec::MouseClick { button } => {
                let btn = MouseBtn::from_name(button);
                vec![
                    Self::MouseButton {
                        button: btn,
                        pressed: true,
                    },
                    Self::MouseButton {
                        button: btn,
                        pressed: false,
                    },
                ]
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct FsmStepOutput {
    pub state: FsmState,
    pub events: Vec<OsActionEvent>,
    pub active_action_label: String,
    pub pinch_active: bool,
    pub volume_level_hud: Option<f32>,
    pub cursor_frozen: bool,
}
