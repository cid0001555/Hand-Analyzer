use std::time::{Duration, Instant};
use glam::Vec3A;

use crate::classifier::GestureTag;
use crate::config::{AppConfig, ContinuousMapping, GestureMappingConfig};
use crate::fsm::state::{
    ContinuousMode, FsmState, FsmStepOutput, MouseBtn, OsActionEvent,
};
use crate::math::one_euro::OneEuroFilterVec3A;
use crate::math::OneEuroFilterScalar;

const DISCRETE_MIN_CONFIDENCE: f32 = 0.84;
const ANTI_DIP_DIST_VEL_THRESHOLD: f32 = -0.18;
const ANTI_DIP_ANGLE_VEL_THRESHOLD: f32 = -45.0;
const RELEASE_DIST_VEL_THRESHOLD: f32 = 0.20;
const RELEASE_ANGLE_VEL_THRESHOLD: f32 = 50.0;
const POINTER_BASE_PIXELS: f32 = 1920.0;

#[derive(Debug, Clone)]
pub struct FrameObservation {
    pub now: Instant,
    pub dt_sec: f32,
    pub tag: GestureTag,
    pub confidence: f32,
    pub landmarks: [Vec3A; 21],
    pub pinch_index_dist: f32,
    pub pinch_aperture_deg: f32,
    pub pinch_middle_dist: f32,
    pub middle_aperture_deg: f32,
    pub middle_pip_deg: f32,
}

pub struct StateMachine {
    state: FsmState,
    pointer_anchor_filter: OneEuroFilterVec3A,
    pinch_vel_filter: OneEuroFilterScalar,
    aperture_vel_filter: OneEuroFilterScalar,
    prev_pointer_pos: Option<Vec3A>,
    pointer_rem_x: f32,
    pointer_rem_y: f32,
    click_lock_anchor: Option<Vec3A>,
    drag_qualified: bool,
    drag_unlocked: bool,
    awaiting_click_rearm: bool,
    prev_scroll_pos: Option<Vec3A>,
    scroll_accum_x: f32,
    scroll_accum_y: f32,
    right_pinch_active: bool,
    freeze_until: Option<Instant>,
    last_sent_volume: f32,
    clutch_dwell_sec: f32,
    discrete_candidate: Option<GestureTag>,
    discrete_dwell_sec: f32,
}

impl StateMachine {
    pub fn new(require_clutch: bool) -> Self {
        let initial_state = if require_clutch {
            FsmState::Locked {
                unlock_progress: 0.0,
                unlock_started: None,
            }
        } else {
            FsmState::Armed
        };

        Self {
            state: initial_state,
            pointer_anchor_filter: OneEuroFilterVec3A::new(1.15, 0.35, 1.0),
            pinch_vel_filter: OneEuroFilterScalar::new(1.5, 0.05, 1.0),
            aperture_vel_filter: OneEuroFilterScalar::new(1.5, 0.05, 1.0),
            prev_pointer_pos: None,
            pointer_rem_x: 0.0,
            pointer_rem_y: 0.0,
            click_lock_anchor: None,
            drag_qualified: false,
            drag_unlocked: false,
            awaiting_click_rearm: false,
            prev_scroll_pos: None,
            scroll_accum_x: 0.0,
            scroll_accum_y: 0.0,
            right_pinch_active: false,
            freeze_until: None,
            last_sent_volume: 0.50,
            clutch_dwell_sec: 0.0,
            discrete_candidate: None,
            discrete_dwell_sec: 0.0,
        }
    }

    #[inline]
    pub fn state(&self) -> &FsmState {
        &self.state
    }

    pub fn on_hand_lost(&mut self, now: Instant) -> FsmStepOutput {
        let mut events = Vec::new();
        self.reset_continuous_trackers(&mut events);
        self.clutch_dwell_sec = 0.0;
        self.discrete_candidate = None;
        self.discrete_dwell_sec = 0.0;

        match &mut self.state {
            FsmState::Locked {
                unlock_progress,
                unlock_started,
            } => {
                *unlock_progress = 0.0;
                *unlock_started = None;
            }
            FsmState::Dwelling { .. } | FsmState::Continuous { .. } => {
                self.state = FsmState::Armed;
            }
            FsmState::Cooldown { until, .. } => {
                if now >= *until {
                    self.state = FsmState::Armed;
                }
            }
            FsmState::Armed => {}
        }

        FsmStepOutput {
            state: self.state.clone(),
            events,
            active_action_label: "NO_HAND".to_string(),
            pinch_active: false,
            volume_level_hud: None,
            cursor_frozen: false,
        }
    }

    pub fn update(&mut self, obs: &FrameObservation, cfg: &AppConfig) -> FsmStepOutput {
        let mut events = Vec::new();
        let dt = obs.dt_sec.clamp(0.001, 0.15);
        let clutch_tag = GestureTag::from_name(&cfg.fsm.clutch_gesture);
        let clutch_target_sec = (cfg.fsm.clutch_dwell_ms.max(50) as f32) * 0.001;

        let (_filtered_pinch, pinch_velocity) = self
            .pinch_vel_filter
            .filter_with_velocity(obs.pinch_index_dist, dt);
        let (_filtered_angle, aperture_velocity) = self
            .aperture_vel_filter
            .filter_with_velocity(obs.pinch_aperture_deg, dt);

        if let FsmState::Locked {
            mut unlock_started, ..
        } = self.state.clone()
        {
            if !cfg.fsm.require_clutch {
                self.state = FsmState::Armed;
            } else {
                if obs.tag == clutch_tag && obs.confidence >= DISCRETE_MIN_CONFIDENCE {
                    let _ = unlock_started.get_or_insert(obs.now);
                    self.clutch_dwell_sec += dt;
                } else {
                    self.clutch_dwell_sec = (self.clutch_dwell_sec - dt * 2.5).max(0.0);
                    if self.clutch_dwell_sec == 0.0 {
                        unlock_started = None;
                    }
                }

                let unlock_progress = (self.clutch_dwell_sec / clutch_target_sec).clamp(0.0, 1.0);
                if unlock_progress >= 1.0 {
                    self.clutch_dwell_sec = 0.0;
                    self.state = FsmState::Cooldown {
                        last_gesture: clutch_tag,
                        until: obs.now + Duration::from_millis(350),
                        require_neutral_reset: true,
                    };
                    return FsmStepOutput {
                        state: self.state.clone(),
                        events,
                        active_action_label: "UNLOCKED".to_string(),
                        pinch_active: false,
                        volume_level_hud: None,
                        cursor_frozen: false,
                    };
                }

                self.state = FsmState::Locked {
                    unlock_progress,
                    unlock_started,
                };
                return FsmStepOutput {
                    state: self.state.clone(),
                    events,
                    active_action_label: format!("Hold {} to arm", cfg.fsm.clutch_gesture),
                    pinch_active: false,
                    volume_level_hud: None,
                    cursor_frozen: false,
                };
            }
        }

        if cfg.fsm.require_clutch && !matches!(self.state, FsmState::Cooldown { .. }) {
            if obs.tag == clutch_tag && obs.confidence >= DISCRETE_MIN_CONFIDENCE {
                self.clutch_dwell_sec += dt;
                let lock_prog = (self.clutch_dwell_sec / clutch_target_sec).clamp(0.0, 1.0);

                if lock_prog >= 1.0 {
                    self.clutch_dwell_sec = 0.0;
                    self.reset_continuous_trackers(&mut events);
                    self.state = FsmState::Locked {
                        unlock_progress: 0.0,
                        unlock_started: None,
                    };
                    return FsmStepOutput {
                        state: self.state.clone(),
                        events,
                        active_action_label: "LOCKED".to_string(),
                        pinch_active: false,
                        volume_level_hud: None,
                        cursor_frozen: false,
                    };
                } else {
                    self.reset_continuous_trackers(&mut events);
                    let dwell_state = FsmState::Dwelling {
                        candidate: clutch_tag.clone(),
                        started_at: obs.now,
                        progress: lock_prog,
                    };
                    self.state = dwell_state.clone();
                    return FsmStepOutput {
                        state: dwell_state,
                        events,
                        active_action_label: "Locking".to_string(),
                        pinch_active: false,
                        volume_level_hud: None,
                        cursor_frozen: false,
                    };
                }
            } else {
                self.clutch_dwell_sec = (self.clutch_dwell_sec - dt * 2.5).max(0.0);
            }
        }

        if let FsmState::Cooldown {
            last_gesture,
            until,
            require_neutral_reset,
        } = self.state.clone()
        {
            let timer_expired = obs.now >= until;
            let gesture_released = !require_neutral_reset || obs.tag != last_gesture;

            if timer_expired && gesture_released {
                self.state = FsmState::Armed;
            } else {
                return FsmStepOutput {
                    state: self.state.clone(),
                    events,
                    active_action_label: format!("Cooldown ({last_gesture})"),
                    pinch_active: false,
                    volume_level_hud: None,
                    cursor_frozen: false,
                };
            }
        }

        if let FsmState::Continuous {
            mode: ContinuousMode::MousePointer,
            pinch_active,
            ..
        } = &self.state
        {
            let close_click_motion = matches!(obs.tag, GestureTag::Fist | GestureTag::Neutral)
                && obs.pinch_index_dist < cfg.fsm.click_exit_threshold * 1.12
                && obs.pinch_aperture_deg < cfg.fsm.click_exit_angle_deg * 1.12;

            if obs.tag != GestureTag::Pointing && (*pinch_active || close_click_motion) {
                if let Some(GestureMappingConfig::Continuous(ptr_cfg)) =
                    cfg.mappings.get(GestureTag::Pointing.as_key())
                {
                    self.discrete_candidate = None;
                    self.discrete_dwell_sec = 0.0;
                    return self.step_continuous(
                        obs,
                        cfg,
                        ptr_cfg,
                        pinch_velocity,
                        aperture_velocity,
                        &mut events,
                    );
                }
            }
        }

        let mapping_opt = cfg.mappings.get(obs.tag.as_key());

        match mapping_opt {
            Some(GestureMappingConfig::Continuous(cont_cfg)) => {
                self.discrete_candidate = None;
                self.discrete_dwell_sec = 0.0;
                self.step_continuous(
                    obs,
                    cfg,
                    cont_cfg,
                    pinch_velocity,
                    aperture_velocity,
                    &mut events,
                )
            }
            Some(GestureMappingConfig::Discrete(disc_cfg))
                if obs.confidence >= DISCRETE_MIN_CONFIDENCE =>
            {
                self.reset_continuous_trackers(&mut events);
                let dwell_sec =
                    (disc_cfg.dwell_ms.unwrap_or(cfg.fsm.default_dwell_ms).max(1) as f32) * 0.001;
                let cooldown_ms = disc_cfg
                    .cooldown_ms
                    .unwrap_or(cfg.fsm.default_cooldown_ms)
                    .max(50);
                let label = disc_cfg.summary_label();

                if self.discrete_candidate.as_ref() == Some(&obs.tag) {
                    self.discrete_dwell_sec += dt;
                } else {
                    self.discrete_candidate = Some(obs.tag.clone());
                    self.discrete_dwell_sec = dt;
                }

                let progress = (self.discrete_dwell_sec / dwell_sec).clamp(0.0, 1.0);

                if progress >= 1.0 {
                    self.discrete_candidate = None;
                    self.discrete_dwell_sec = 0.0;
                    if let Some(action_spec) = disc_cfg.effective_action() {
                        events.extend(OsActionEvent::from_action_spec(action_spec));
                    }
                    self.state = FsmState::Cooldown {
                        last_gesture: obs.tag.clone(),
                        until: obs.now + Duration::from_millis(cooldown_ms),
                        require_neutral_reset: true,
                    };
                } else {
                    self.state = FsmState::Dwelling {
                        candidate: obs.tag.clone(),
                        started_at: obs.now,
                        progress,
                    };
                }

                FsmStepOutput {
                    state: self.state.clone(),
                    events,
                    active_action_label: label,
                    pinch_active: false,
                    volume_level_hud: None,
                    cursor_frozen: false,
                }
            }
            _ => {
                if let FsmState::Continuous {
                    mode: ContinuousMode::MousePointer,
                    pinch_active: true,
                    ..
                } = &self.state
                {
                    if let Some(GestureMappingConfig::Continuous(ptr_cfg)) =
                        cfg.mappings.get(GestureTag::Pointing.as_key())
                    {
                        return self.step_continuous(
                            obs,
                            cfg,
                            ptr_cfg,
                            pinch_velocity,
                            aperture_velocity,
                            &mut events,
                        );
                    }
                }

                if let Some(ref cand) = self.discrete_candidate.clone() {
                    self.discrete_dwell_sec = (self.discrete_dwell_sec - dt * 2.5).max(0.0);
                    if self.discrete_dwell_sec > 0.0 {
                        if let Some(GestureMappingConfig::Discrete(disc_cfg)) =
                            cfg.mappings.get(cand.as_key())
                        {
                            let dwell_sec = (disc_cfg
                                .dwell_ms
                                .unwrap_or(cfg.fsm.default_dwell_ms)
                                .max(1) as f32)
                                * 0.001;
                            let progress = (self.discrete_dwell_sec / dwell_sec).clamp(0.0, 1.0);
                            let dwell_state = FsmState::Dwelling {
                                candidate: cand.clone(),
                                started_at: obs.now,
                                progress,
                            };
                            self.state = dwell_state.clone();
                            return FsmStepOutput {
                                state: dwell_state,
                                events,
                                active_action_label: disc_cfg.summary_label(),
                                pinch_active: false,
                                volume_level_hud: None,
                                cursor_frozen: false,
                            };
                        }
                    }
                    self.discrete_candidate = None;
                }

                self.reset_continuous_trackers(&mut events);
                self.state = FsmState::Armed;
                FsmStepOutput {
                    state: FsmState::Armed,
                    events,
                    active_action_label: "IDLE".to_string(),
                    pinch_active: false,
                    volume_level_hud: None,
                    cursor_frozen: false,
                }
            }
        }
    }

    fn step_continuous(
        &mut self,
        obs: &FrameObservation,
        cfg: &AppConfig,
        cont_cfg: &ContinuousMapping,
        pinch_velocity: f32,
        aperture_velocity: f32,
        events: &mut Vec<OsActionEvent>,
    ) -> FsmStepOutput {
        let target_mode: ContinuousMode = cont_cfg.control_type.into();
        let raw_metacarpal_anchor = 0.70 * obs.landmarks[5] + 0.30 * obs.landmarks[9];

        let (initial_anchor, initial_value, prev_pinch) = match &self.state {
            FsmState::Continuous {
                mode,
                initial_anchor,
                initial_value,
                pinch_active,
            } if *mode == target_mode => (*initial_anchor, *initial_value, *pinch_active),
            _ => {
                self.reset_continuous_trackers(events);
                (raw_metacarpal_anchor, self.last_sent_volume, false)
            }
        };

        let mut pinch_active = prev_pinch;
        let mut volume_hud = None;
        let mut cursor_frozen = false;

        match target_mode {
            ContinuousMode::MousePointer => {
                let pointer_pos = self
                    .pointer_anchor_filter
                    .filter(raw_metacarpal_anchor, obs.dt_sec);

                let opened_past_click = obs.pinch_aperture_deg > cfg.fsm.click_exit_angle_deg
                    || obs.pinch_index_dist > cfg.fsm.click_exit_threshold;
                if opened_past_click {
                    self.awaiting_click_rearm = false;
                }

                let enter_click = !self.awaiting_click_rearm
                    && obs.pinch_aperture_deg < cfg.fsm.click_enter_angle_deg
                    && obs.pinch_index_dist < cfg.fsm.click_enter_threshold;

                let enter_firm_drag = obs.pinch_aperture_deg < cfg.fsm.pinch_enter_angle_deg
                    && obs.pinch_index_dist < cfg.fsm.pinch_enter_threshold;

                let exit_active = if self.drag_qualified {
                    obs.pinch_aperture_deg > cfg.fsm.pinch_exit_angle_deg
                        || obs.pinch_index_dist > cfg.fsm.pinch_exit_threshold
                } else {
                    opened_past_click
                };

                if !pinch_active && enter_click {
                    pinch_active = true;
                    self.drag_qualified = enter_firm_drag;
                    self.click_lock_anchor = Some(pointer_pos);
                    self.drag_unlocked = false;
                    events.push(OsActionEvent::MouseButton {
                        button: MouseBtn::Left,
                        pressed: true,
                    });
                } else if pinch_active {
                    if !self.drag_unlocked && enter_firm_drag {
                        self.drag_qualified = true;
                    }

                    if exit_active {
                        pinch_active = false;
                        self.drag_qualified = false;
                        self.click_lock_anchor = None;
                        self.drag_unlocked = false;
                        self.extend_freeze(obs.now, Duration::from_millis(85));
                        events.push(OsActionEvent::MouseButton {
                            button: MouseBtn::Left,
                            pressed: false,
                        });
                    }
                }

                let right_enter = obs.middle_pip_deg < 55.0
                    && obs.pinch_middle_dist < cfg.fsm.click_enter_threshold
                    && obs.middle_aperture_deg < cfg.fsm.click_enter_angle_deg;
                let right_exit = obs.pinch_middle_dist > cfg.fsm.click_exit_threshold
                    || obs.middle_aperture_deg > cfg.fsm.click_exit_angle_deg
                    || obs.middle_pip_deg > 65.0;

                if !self.right_pinch_active && !pinch_active && right_enter {
                    self.right_pinch_active = true;
                    self.extend_freeze(obs.now, Duration::from_millis(120));
                    events.push(OsActionEvent::MouseButton {
                        button: MouseBtn::Right,
                        pressed: true,
                    });
                    events.push(OsActionEvent::MouseButton {
                        button: MouseBtn::Right,
                        pressed: false,
                    });
                } else if self.right_pinch_active && right_exit {
                    self.right_pinch_active = false;
                }

                let in_pinch_zone = obs.pinch_index_dist < cfg.fsm.click_exit_threshold * 1.08;
                let rapid_closing = !self.drag_unlocked
                    && in_pinch_zone
                    && (pinch_velocity < ANTI_DIP_DIST_VEL_THRESHOLD
                        || aperture_velocity < ANTI_DIP_ANGLE_VEL_THRESHOLD);
                let rapid_opening = pinch_active
                    && (pinch_velocity > RELEASE_DIST_VEL_THRESHOLD
                        || aperture_velocity > RELEASE_ANGLE_VEL_THRESHOLD);

                if rapid_closing && cont_cfg.freeze_on_click_ms > 0 {
                    self.extend_freeze(
                        obs.now,
                        Duration::from_millis(cont_cfg.freeze_on_click_ms),
                    );
                } else if rapid_opening {
                    self.extend_freeze(obs.now, Duration::from_millis(85));
                }

                let time_frozen = self
                    .freeze_until
                    .map(|until| obs.now < until)
                    .unwrap_or(false);

                let mut click_deadband_locked = false;
                if pinch_active && !self.drag_unlocked {
                    if let Some(origin) = self.click_lock_anchor {
                        let d_move = ((pointer_pos.x - origin.x).powi(2)
                            + (pointer_pos.y - origin.y).powi(2))
                        .sqrt();
                        if d_move >= cont_cfg.drag_deadzone {
                            if self.drag_qualified {
                                self.drag_unlocked = true;
                            } else {
                                pinch_active = false;
                                self.click_lock_anchor = None;
                                self.awaiting_click_rearm = true;
                                events.push(OsActionEvent::MouseButton {
                                    button: MouseBtn::Left,
                                    pressed: false,
                                });
                            }
                        } else {
                            click_deadband_locked = true;
                        }
                    }
                }

                let brake_factor = if !pinch_active
                    && !self.awaiting_click_rearm
                    && obs.pinch_index_dist < cfg.fsm.click_exit_threshold
                    && obs.pinch_aperture_deg < cfg.fsm.click_exit_angle_deg
                {
                    let angle_span =
                        (cfg.fsm.click_exit_angle_deg - cfg.fsm.click_enter_angle_deg).max(3.0);
                    let t_angle = ((obs.pinch_aperture_deg - cfg.fsm.click_enter_angle_deg)
                        / angle_span)
                        .clamp(0.0, 1.0);

                    let dist_span =
                        (cfg.fsm.click_exit_threshold - cfg.fsm.click_enter_threshold).max(0.006);
                    let t_dist = ((obs.pinch_index_dist - cfg.fsm.click_enter_threshold)
                        / dist_span)
                        .clamp(0.0, 1.0);

                    let combined = t_angle.max(t_dist);
                    combined * combined
                } else {
                    1.0
                };

                cursor_frozen = time_frozen || click_deadband_locked || brake_factor < 0.08;

                if let Some(prev_pos) = self.prev_pointer_pos {
                    if !time_frozen && !click_deadband_locked && brake_factor >= 0.02 {
                        let delta = pointer_pos - prev_pos;
                        let radial = (delta.x * delta.x + delta.y * delta.y).sqrt();
                        if radial > cont_cfg.deadzone {
                            let hand_speed = radial / obs.dt_sec.max(0.004);
                            let speed_t = ((hand_speed - 0.05) / 0.45).clamp(0.0, 1.0);
                            let ballistic_mult = 0.85 + 0.95 * speed_t;

                            let scale = POINTER_BASE_PIXELS
                                * cont_cfg.sensitivity
                                * ballistic_mult
                                * brake_factor;

                            self.pointer_rem_x += delta.x * scale;
                            self.pointer_rem_y += delta.y * scale;

                            let dx = self.pointer_rem_x.trunc() as i32;
                            let dy = self.pointer_rem_y.trunc() as i32;

                            if dx != 0 || dy != 0 {
                                self.pointer_rem_x -= dx as f32;
                                self.pointer_rem_y -= dy as f32;
                                events.push(OsActionEvent::MovePointerRel { dx, dy });
                            }
                        }
                    } else {
                        self.pointer_rem_x = 0.0;
                        self.pointer_rem_y = 0.0;
                    }
                }
                self.prev_pointer_pos = Some(pointer_pos);
            }

            ContinuousMode::ScrollWheel => {
                let midpoint = 0.5 * (obs.landmarks[8] + obs.landmarks[12]);
                if let Some(prev) = self.prev_scroll_pos {
                    let delta = midpoint - prev;
                    self.scroll_accum_y += (-delta.y) * cont_cfg.scroll_speed_y * 12.0;
                    self.scroll_accum_x += delta.x * cont_cfg.scroll_speed_x * 12.0;

                    let wheel_y = self.scroll_accum_y.trunc() as i32;
                    let wheel_x = self.scroll_accum_x.trunc() as i32;

                    if wheel_y != 0 || wheel_x != 0 {
                        self.scroll_accum_y -= wheel_y as f32;
                        self.scroll_accum_x -= wheel_x as f32;
                        events.push(OsActionEvent::MouseScroll { wheel_y, wheel_x });
                    }
                }
                self.prev_scroll_pos = Some(midpoint);
            }

            ContinuousMode::VolumeSlider => {
                let y = obs.landmarks[9].y;
                let y_min = cont_cfg.vertical_range_min;
                let y_max = (cont_cfg.vertical_range_max).max(y_min + 0.05);
                let normalized_vol = (1.0 - (y - y_min) / (y_max - y_min)).clamp(0.0, 1.0);
                volume_hud = Some(normalized_vol);

                if (normalized_vol - self.last_sent_volume).abs() >= 0.008 {
                    self.last_sent_volume = normalized_vol;
                    events.push(OsActionEvent::SetSystemVolume(normalized_vol));
                }
            }
        }

        let new_state = FsmState::Continuous {
            mode: target_mode,
            initial_anchor,
            initial_value,
            pinch_active,
        };
        self.state = new_state.clone();

        let base_label = if !cont_cfg.description.is_empty() {
            target_mode.label().to_string()
        } else {
            format!("Continuous::{:?}", target_mode)
        };
        let label = if target_mode == ContinuousMode::MousePointer && pinch_active {
            if self.drag_unlocked {
                format!("{base_label} [DRAG]")
            } else {
                format!("{base_label} [CLICK]")
            }
        } else {
            base_label
        };

        FsmStepOutput {
            state: new_state,
            events: std::mem::take(events),
            active_action_label: label,
            pinch_active,
            volume_level_hud: volume_hud,
            cursor_frozen,
        }
    }

    #[inline]
    fn extend_freeze(&mut self, now: Instant, duration: Duration) {
        let candidate_until = now + duration;
        self.freeze_until = Some(match self.freeze_until {
            Some(existing) if existing > candidate_until => existing,
            _ => candidate_until,
        });
    }

    fn reset_continuous_trackers(&mut self, events: &mut Vec<OsActionEvent>) {
        if let FsmState::Continuous {
            pinch_active: true, ..
        } = &self.state
        {
            events.push(OsActionEvent::MouseButton {
                button: MouseBtn::Left,
                pressed: false,
            });
        }
        self.pointer_anchor_filter.reset();
        self.prev_pointer_pos = None;
        self.pointer_rem_x = 0.0;
        self.pointer_rem_y = 0.0;
        self.click_lock_anchor = None;
        self.drag_qualified = false;
        self.drag_unlocked = false;
        self.awaiting_click_rearm = false;
        self.prev_scroll_pos = None;
        self.scroll_accum_x = 0.0;
        self.scroll_accum_y = 0.0;
        self.right_pinch_active = false;
        self.freeze_until = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::config::{
        ActionSpec, ContinuousControlType, DiscreteMapping, FilterConfig, FsmConfig, GeneralConfig,
        MediaCommandConfig,
    };

    fn test_config() -> AppConfig {
        let mut mappings = HashMap::new();
        mappings.insert(
            "OPEN_PALM".to_string(),
            GestureMappingConfig::Discrete(DiscreteMapping {
                description: "PlayPause".to_string(),
                dwell_ms: Some(250),
                cooldown_ms: Some(500),
                action: Some(ActionSpec::Media {
                    command: MediaCommandConfig::PlayPause,
                }),
                action_linux: None,
                action_windows: None,
            }),
        );
        mappings.insert(
            "FIST".to_string(),
            GestureMappingConfig::Discrete(DiscreteMapping {
                description: "Mute".to_string(),
                dwell_ms: Some(300),
                cooldown_ms: Some(800),
                action: Some(ActionSpec::Media {
                    command: MediaCommandConfig::Mute,
                }),
                action_linux: None,
                action_windows: None,
            }),
        );
        mappings.insert(
            "POINTING".to_string(),
            GestureMappingConfig::Continuous(ContinuousMapping {
                description: "Air Mouse".to_string(),
                control_type: ContinuousControlType::MousePointer,
                sensitivity: 3.0,
                deadzone: 0.0004,
                drag_deadzone: 0.010,
                freeze_on_click_ms: 140,
                scroll_speed_y: 12.0,
                scroll_speed_x: 8.0,
                vertical_range_min: 0.2,
                vertical_range_max: 0.8,
            }),
        );

        AppConfig {
            general: GeneralConfig::default(),
            filter: FilterConfig::default(),
            fsm: FsmConfig {
                require_clutch: true,
                clutch_gesture: "SHAKA".to_string(),
                clutch_dwell_ms: 400,
                default_dwell_ms: 250,
                default_cooldown_ms: 500,
                click_enter_threshold: 0.046,
                click_exit_threshold: 0.054,
                click_enter_angle_deg: 29.0,
                click_exit_angle_deg: 35.0,
                pinch_enter_threshold: 0.033,
                pinch_exit_threshold: 0.046,
                pinch_enter_angle_deg: 19.0,
                pinch_exit_angle_deg: 27.0,
            },
            mappings,
        }
    }

    #[test]
    fn test_fsm_click_and_drag() {
        let cfg = test_config();
        let mut fsm = StateMachine::new(true);
        let t0 = Instant::now();
        let mut lm = [Vec3A::splat(0.5); 21];

        let out = fsm.update(
            &FrameObservation {
                now: t0,
                dt_sec: 0.016,
                tag: GestureTag::OpenPalm,
                confidence: 0.96,
                landmarks: lm,
                pinch_index_dist: 0.082,
                pinch_aperture_deg: 72.0,
                pinch_middle_dist: 0.025,
                middle_aperture_deg: 12.0,
                middle_pip_deg: 95.0,
            },
            &cfg,
        );
        assert!(matches!(out.state, FsmState::Locked { .. }));
        assert!(out.events.is_empty());

        for step in 1..=10 {
            let _ = fsm.update(
                &FrameObservation {
                    now: t0 + Duration::from_millis(step * 45),
                    dt_sec: 0.045,
                    tag: GestureTag::Shaka,
                    confidence: 0.95,
                    landmarks: lm,
                    pinch_index_dist: 0.082,
                    pinch_aperture_deg: 72.0,
                    pinch_middle_dist: 0.025,
                    middle_aperture_deg: 12.0,
                    middle_pip_deg: 95.0,
                },
                &cfg,
            );
        }
        assert!(matches!(fsm.state(), FsmState::Cooldown { .. }));

        let armed = fsm.update(
            &FrameObservation {
                now: t0 + Duration::from_millis(900),
                dt_sec: 0.016,
                tag: GestureTag::Neutral,
                confidence: 0.5,
                landmarks: lm,
                pinch_index_dist: 0.082,
                pinch_aperture_deg: 72.0,
                pinch_middle_dist: 0.025,
                middle_aperture_deg: 12.0,
                middle_pip_deg: 95.0,
            },
            &cfg,
        );
        assert_eq!(armed.state, FsmState::Armed);

        let aiming = fsm.update(
            &FrameObservation {
                now: t0 + Duration::from_millis(920),
                dt_sec: 0.016,
                tag: GestureTag::Pointing,
                confidence: 0.95,
                landmarks: lm,
                pinch_index_dist: 0.080,
                pinch_aperture_deg: 74.0,
                pinch_middle_dist: 0.020,
                middle_aperture_deg: 10.0,
                middle_pip_deg: 95.0,
            },
            &cfg,
        );
        assert!(!aiming.pinch_active);
        assert!(!aiming.cursor_frozen);
        assert!(aiming.events.is_empty());

        let light_click = fsm.update(
            &FrameObservation {
                now: t0 + Duration::from_millis(940),
                dt_sec: 0.016,
                tag: GestureTag::Fist,
                confidence: 0.94,
                landmarks: lm,
                pinch_index_dist: 0.041,
                pinch_aperture_deg: 25.0,
                pinch_middle_dist: 0.060,
                middle_aperture_deg: 45.0,
                middle_pip_deg: 95.0,
            },
            &cfg,
        );
        assert!(light_click.pinch_active);
        assert_eq!(light_click.active_action_label, "AIR_MOUSE [CLICK]");
        assert!(matches!(
            light_click.events[0],
            OsActionEvent::MouseButton {
                button: MouseBtn::Left,
                pressed: true
            }
        ));

        lm[5] = Vec3A::new(0.60, 0.60, 0.5);
        lm[9] = Vec3A::new(0.60, 0.60, 0.5);
        let mut auto_released = false;
        for step in 1..=6 {
            let out_move = fsm.update(
                &FrameObservation {
                    now: t0 + Duration::from_millis(940 + step * 16),
                    dt_sec: 0.016,
                    tag: GestureTag::Pointing,
                    confidence: 0.95,
                    landmarks: lm,
                    pinch_index_dist: 0.041,
                    pinch_aperture_deg: 25.0,
                    pinch_middle_dist: 0.060,
                    middle_aperture_deg: 45.0,
                    middle_pip_deg: 95.0,
                },
                &cfg,
            );
            if out_move.events.iter().any(|e| {
                matches!(
                    e,
                    OsActionEvent::MouseButton {
                        button: MouseBtn::Left,
                        pressed: false
                    }
                )
            }) {
                auto_released = true;
                break;
            }
        }
        assert!(auto_released);

        let _ = fsm.update(
            &FrameObservation {
                now: t0 + Duration::from_millis(1100),
                dt_sec: 0.016,
                tag: GestureTag::Pointing,
                confidence: 0.95,
                landmarks: lm,
                pinch_index_dist: 0.070,
                pinch_aperture_deg: 55.0,
                pinch_middle_dist: 0.060,
                middle_aperture_deg: 45.0,
                middle_pip_deg: 95.0,
            },
            &cfg,
        );

        let firm_down = fsm.update(
            &FrameObservation {
                now: t0 + Duration::from_millis(1250),
                dt_sec: 0.016,
                tag: GestureTag::Pointing,
                confidence: 0.95,
                landmarks: lm,
                pinch_index_dist: 0.024,
                pinch_aperture_deg: 11.0,
                pinch_middle_dist: 0.060,
                middle_aperture_deg: 45.0,
                middle_pip_deg: 95.0,
            },
            &cfg,
        );
        assert!(firm_down.pinch_active);

        lm[5] = Vec3A::new(0.72, 0.72, 0.5);
        lm[9] = Vec3A::new(0.72, 0.72, 0.5);
        let mut drag_engaged = false;
        for step in 1..=6 {
            let drag_step = fsm.update(
                &FrameObservation {
                    now: t0 + Duration::from_millis(1420 + step * 16),
                    dt_sec: 0.016,
                    tag: GestureTag::Pointing,
                    confidence: 0.95,
                    landmarks: lm,
                    pinch_index_dist: 0.038,
                    pinch_aperture_deg: 20.0,
                    pinch_middle_dist: 0.060,
                    middle_aperture_deg: 45.0,
                    middle_pip_deg: 95.0,
                },
                &cfg,
            );
            if drag_step.active_action_label == "AIR_MOUSE [DRAG]" {
                drag_engaged = true;
            }
        }
        assert!(drag_engaged);
    }
}
