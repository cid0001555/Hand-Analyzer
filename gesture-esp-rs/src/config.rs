use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use notify::{Config as NotifyConfig, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::error::{DomainResult, GestureEspError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub general: GeneralConfig,
    #[serde(default)]
    pub filter: FilterConfig,
    #[serde(default)]
    pub fsm: FsmConfig,
    #[serde(default)]
    pub mappings: HashMap<String, GestureMappingConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    #[serde(default = "default_camera_index")]
    pub camera_index: u32,
    #[serde(default = "default_capture_width")]
    pub capture_width: u32,
    #[serde(default = "default_capture_height")]
    pub capture_height: u32,
    #[serde(default = "default_target_fps")]
    pub target_fps: u32,
    #[serde(default = "default_true")]
    pub mirror_horizontal: bool,
    #[serde(default = "default_exec_provider")]
    pub execution_provider: String,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            camera_index: default_camera_index(),
            capture_width: default_capture_width(),
            capture_height: default_capture_height(),
            target_fps: default_target_fps(),
            mirror_horizontal: true,
            execution_provider: default_exec_provider(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FilterConfig {
    #[serde(default = "default_min_cutoff")]
    pub min_cutoff: f32,
    #[serde(default = "default_beta")]
    pub beta: f32,
    #[serde(default = "default_d_cutoff")]
    pub d_cutoff: f32,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            min_cutoff: default_min_cutoff(),
            beta: default_beta(),
            d_cutoff: default_d_cutoff(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsmConfig {
    #[serde(default = "default_true")]
    pub require_clutch: bool,
    #[serde(default = "default_clutch_gesture")]
    pub clutch_gesture: String,
    #[serde(default = "default_clutch_dwell_ms")]
    pub clutch_dwell_ms: u64,
    #[serde(default = "default_dwell_ms")]
    pub default_dwell_ms: u64,
    #[serde(default = "default_cooldown_ms")]
    pub default_cooldown_ms: u64,
    #[serde(default = "default_click_enter")]
    pub click_enter_threshold: f32,
    #[serde(default = "default_click_exit")]
    pub click_exit_threshold: f32,
    #[serde(default = "default_click_enter_angle")]
    pub click_enter_angle_deg: f32,
    #[serde(default = "default_click_exit_angle")]
    pub click_exit_angle_deg: f32,
    #[serde(default = "default_pinch_enter")]
    pub pinch_enter_threshold: f32,
    #[serde(default = "default_pinch_exit")]
    pub pinch_exit_threshold: f32,
    #[serde(default = "default_pinch_enter_angle")]
    pub pinch_enter_angle_deg: f32,
    #[serde(default = "default_pinch_exit_angle")]
    pub pinch_exit_angle_deg: f32,
}

impl Default for FsmConfig {
    fn default() -> Self {
        Self {
            require_clutch: true,
            clutch_gesture: default_clutch_gesture(),
            clutch_dwell_ms: default_clutch_dwell_ms(),
            default_dwell_ms: default_dwell_ms(),
            default_cooldown_ms: default_cooldown_ms(),
            click_enter_threshold: default_click_enter(),
            click_exit_threshold: default_click_exit(),
            click_enter_angle_deg: default_click_enter_angle(),
            click_exit_angle_deg: default_click_exit_angle(),
            pinch_enter_threshold: default_pinch_enter(),
            pinch_exit_threshold: default_pinch_exit(),
            pinch_enter_angle_deg: default_pinch_enter_angle(),
            pinch_exit_angle_deg: default_pinch_exit_angle(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
pub enum GestureMappingConfig {
    Discrete(DiscreteMapping),
    Continuous(ContinuousMapping),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteMapping {
    #[serde(default)]
    pub description: String,
    pub dwell_ms: Option<u64>,
    pub cooldown_ms: Option<u64>,
    pub action: Option<ActionSpec>,
    pub action_linux: Option<ActionSpec>,
    pub action_windows: Option<ActionSpec>,
}

impl DiscreteMapping {
    pub fn effective_action(&self) -> Option<&ActionSpec> {
        #[cfg(target_os = "linux")]
        {
            self.action_linux.as_ref().or(self.action.as_ref())
        }
        #[cfg(target_os = "windows")]
        {
            self.action_windows.as_ref().or(self.action.as_ref())
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            self.action
                .as_ref()
                .or(self.action_linux.as_ref())
                .or(self.action_windows.as_ref())
        }
    }

    pub fn summary_label(&self) -> String {
        if !self.description.is_empty() {
            return self.description.clone();
        }
        match self.effective_action() {
            Some(ActionSpec::Media { command }) => format!("Media::{command:?}"),
            Some(ActionSpec::Hotkey { keys }) => format!("Hotkey({})", keys.join("+")),
            Some(ActionSpec::Shell { command }) => format!("Shell({command})"),
            Some(ActionSpec::MouseClick { button }) => format!("Click({button})"),
            None => "Unbound".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ActionSpec {
    Media { command: MediaCommandConfig },
    Hotkey { keys: Vec<String> },
    Shell { command: String },
    MouseClick { button: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaCommandConfig {
    PlayPause,
    Mute,
    NextTrack,
    PrevTrack,
    VolumeUp,
    VolumeDown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousMapping {
    #[serde(default)]
    pub description: String,
    pub control_type: ContinuousControlType,
    #[serde(default = "default_sensitivity")]
    pub sensitivity: f32,
    #[serde(default = "default_deadzone")]
    pub deadzone: f32,
    #[serde(default = "default_drag_deadzone")]
    pub drag_deadzone: f32,
    #[serde(default = "default_freeze_ms")]
    pub freeze_on_click_ms: u64,
    #[serde(default = "default_scroll_y")]
    pub scroll_speed_y: f32,
    #[serde(default = "default_scroll_x")]
    pub scroll_speed_x: f32,
    #[serde(default = "default_vol_min")]
    pub vertical_range_min: f32,
    #[serde(default = "default_vol_max")]
    pub vertical_range_max: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContinuousControlType {
    MousePointer,
    ScrollWheel,
    VolumeSlider,
}

impl AppConfig {
    pub fn load_from_file(path: impl AsRef<Path>) -> DomainResult<Self> {
        let path_ref = path.as_ref();
        let raw = fs::read_to_string(path_ref).map_err(|e| GestureEspError::Config {
            path: path_ref.to_path_buf(),
            message: e.to_string(),
        })?;
        let parsed: AppConfig = toml::from_str(&raw)?;
        parsed.validate(path_ref)?;
        Ok(parsed)
    }

    fn validate(&self, path: &Path) -> DomainResult<()> {
        if self.fsm.click_enter_threshold >= self.fsm.click_exit_threshold {
            return Err(GestureEspError::Config {
                path: path.to_path_buf(),
                message: format!(
                    "click_enter_threshold ({}) must be less than click_exit_threshold ({})",
                    self.fsm.click_enter_threshold, self.fsm.click_exit_threshold
                ),
            });
        }
        if self.fsm.pinch_enter_threshold >= self.fsm.pinch_exit_threshold {
            return Err(GestureEspError::Config {
                path: path.to_path_buf(),
                message: format!(
                    "pinch_enter_threshold ({}) must be less than pinch_exit_threshold ({})",
                    self.fsm.pinch_enter_threshold, self.fsm.pinch_exit_threshold
                ),
            });
        }
        if self.filter.min_cutoff <= 0.0 || self.filter.d_cutoff <= 0.0 {
            return Err(GestureEspError::Config {
                path: path.to_path_buf(),
                message: "filter cutoff values must be > 0.0".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ReloadTelemetry {
    pub generation: u64,
    pub status_text: String,
    pub last_updated: Option<Instant>,
}

pub struct LiveConfigHandle {
    inner: Arc<RwLock<AppConfig>>,
    status: Arc<RwLock<ReloadTelemetry>>,
    generation: Arc<AtomicU64>,
    _watcher: Option<RecommendedWatcher>,
}

impl LiveConfigHandle {
    pub fn initialize(config_path: impl AsRef<Path>) -> DomainResult<Self> {
        let path = config_path.as_ref().to_path_buf();
        let initial_cfg = AppConfig::load_from_file(&path)?;
        let inner = Arc::new(RwLock::new(initial_cfg));
        let generation = Arc::new(AtomicU64::new(1));
        let status = Arc::new(RwLock::new(ReloadTelemetry {
            generation: 1,
            status_text: "SYNCED (v1)".to_string(),
            last_updated: None,
        }));

        let canonical_target = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        let target_filename = canonical_target
            .file_name()
            .map(|s| s.to_os_string())
            .unwrap_or_default();
        let watch_dir = canonical_target
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        let inner_clone = Arc::clone(&inner);
        let status_clone = Arc::clone(&status);
        let gen_clone = Arc::clone(&generation);
        let reload_path = canonical_target.clone();
        let mut last_reload = Instant::now() - Duration::from_secs(2);

        let watcher_result = RecommendedWatcher::new(
            move |res: Result<notify::Event, notify::Error>| match res {
                Ok(event) => {
                    let relevant_kind = matches!(
                        event.kind,
                        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Any
                    );
                    if !relevant_kind {
                        return;
                    }
                    let matches_file = event.paths.iter().any(|p| {
                        p.file_name()
                            .map(|name| name == target_filename)
                            .unwrap_or(false)
                    });
                    if !matches_file {
                        return;
                    }
                    if last_reload.elapsed() < Duration::from_millis(120) {
                        return;
                    }
                    last_reload = Instant::now();

                    match AppConfig::load_from_file(&reload_path) {
                        Ok(new_cfg) => {
                            *inner_clone.write() = new_cfg;
                            let next_gen = gen_clone.fetch_add(1, Ordering::SeqCst) + 1;
                            let mut st = status_clone.write();
                            st.generation = next_gen;
                            st.status_text = format!("RELOADED (v{next_gen})");
                            st.last_updated = Some(Instant::now());
                            info!(
                                generation = next_gen,
                                path = %reload_path.display(),
                                "Configuration reloaded"
                            );
                        }
                        Err(err) => {
                            let mut st = status_clone.write();
                            st.status_text = format!("RELOAD ERR: {err}");
                            st.last_updated = Some(Instant::now());
                            warn!(
                                error = %err,
                                path = %reload_path.display(),
                                "Invalid config reload ignored"
                            );
                        }
                    }
                }
                Err(err) => {
                    warn!(error = %err, "Config watcher error");
                }
            },
            NotifyConfig::default(),
        );

        let watcher = match watcher_result {
            Ok(mut w) => {
                if let Err(err) = w.watch(&watch_dir, RecursiveMode::NonRecursive) {
                    warn!(error = %err, dir = %watch_dir.display(), "Failed to watch config directory");
                    None
                } else {
                    Some(w)
                }
            }
            Err(err) => {
                warn!(error = %err, "Config watcher unavailable");
                None
            }
        };

        Ok(Self {
            inner,
            status,
            generation,
            _watcher: watcher,
        })
    }

    pub fn snapshot(&self) -> AppConfig {
        self.inner.read().clone()
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    pub fn telemetry(&self) -> ReloadTelemetry {
        self.status.read().clone()
    }

    pub fn shared_config(&self) -> Arc<RwLock<AppConfig>> {
        Arc::clone(&self.inner)
    }

    pub fn shared_telemetry(&self) -> Arc<RwLock<ReloadTelemetry>> {
        Arc::clone(&self.status)
    }

    pub fn shared_generation(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.generation)
    }
}

fn default_camera_index() -> u32 {
    0
}
fn default_capture_width() -> u32 {
    1280
}
fn default_capture_height() -> u32 {
    720
}
fn default_target_fps() -> u32 {
    60
}
fn default_true() -> bool {
    true
}
fn default_exec_provider() -> String {
    "auto".to_string()
}
fn default_min_cutoff() -> f32 {
    0.8
}
fn default_beta() -> f32 {
    0.04
}
fn default_d_cutoff() -> f32 {
    1.0
}
fn default_clutch_gesture() -> String {
    "SHAKA".to_string()
}
fn default_clutch_dwell_ms() -> u64 {
    450
}
fn default_dwell_ms() -> u64 {
    250
}
fn default_cooldown_ms() -> u64 {
    750
}
fn default_click_enter() -> f32 {
    0.046
}
fn default_click_exit() -> f32 {
    0.054
}
fn default_click_enter_angle() -> f32 {
    29.0
}
fn default_click_exit_angle() -> f32 {
    35.0
}
fn default_pinch_enter() -> f32 {
    0.033
}
fn default_pinch_exit() -> f32 {
    0.046
}
fn default_pinch_enter_angle() -> f32 {
    19.0
}
fn default_pinch_exit_angle() -> f32 {
    27.0
}
fn default_sensitivity() -> f32 {
    3.0
}
fn default_deadzone() -> f32 {
    0.0004
}
fn default_drag_deadzone() -> f32 {
    0.010
}
fn default_freeze_ms() -> u64 {
    120
}
fn default_scroll_y() -> f32 {
    12.0
}
fn default_scroll_x() -> f32 {
    8.0
}
fn default_vol_min() -> f32 {
    0.20
}
fn default_vol_max() -> f32 {
    0.80
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_mappings_toml_loads_and_validates() {
        let cfg = AppConfig::load_from_file("config/mappings.toml").unwrap();
        assert!(cfg.fsm.require_clutch);
        assert_eq!(cfg.fsm.clutch_gesture, "SHAKA");
        assert!(cfg.mappings.contains_key("OPEN_PALM"));
        assert!(cfg.mappings.contains_key("VICTORY"));
        assert!(cfg.mappings.contains_key("POINTING"));
        assert!(cfg.mappings.contains_key("TWO_FINGER_TOGETHER"));
        assert!(cfg.mappings.contains_key("OK_SIGN_SLIDER"));
    }
}
