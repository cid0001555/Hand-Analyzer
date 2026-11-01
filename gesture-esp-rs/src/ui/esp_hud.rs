use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam_channel::Receiver;
use glam::Vec3A;
use macroquad::color::{Color, WHITE};
use macroquad::input::{is_key_pressed, KeyCode};
use macroquad::math::Rect;
use macroquad::shapes::{
    draw_circle, draw_circle_lines, draw_line, draw_rectangle, draw_rectangle_lines,
};
use macroquad::text::{draw_text_ex, load_ttf_font_from_bytes, Font, TextParams};
use macroquad::texture::{
    draw_texture_ex, DrawTextureParams, FilterMode, Image as MqImage, Texture2D,
};
use macroquad::time::get_fps;
use macroquad::window::{
    clear_background, next_frame, screen_height, screen_width, Conf,
};

use crate::classifier::{GestureTag, RecorderPhase};
use crate::fsm::FsmState;
use crate::ui::theme::TacticalPalette;
use crate::vision::{Handedness, HAND_CONNECTIONS};

const PIP_LANDMARK_INDICES: [usize; 5] = [2, 6, 10, 14, 18];

#[derive(Debug, Clone)]
pub struct HandHudOverlay {
    pub landmarks: [Vec3A; 21],
    pub handedness: Handedness,
    pub presence_score: f32,
    pub pip_angles_deg: [f32; 5],
    pub pinch_index_dist: f32,
    pub pinch_aperture_deg: f32,
    pub used_palm_detector: bool,
}

#[derive(Debug, Clone)]
pub struct HudTelemetryPacket {
    pub frame_rgb: Arc<Vec<u8>>,
    pub frame_width: u32,
    pub frame_height: u32,
    pub frame_sequence: u64,
    pub inference_ms: f32,
    pub pipeline_e2e_ms: f32,
    pub execution_provider: String,
    pub config_status: String,
    pub tag: GestureTag,
    pub confidence: f32,
    pub from_mlp: bool,
    pub fsm_state: FsmState,
    pub action_label: String,
    pub pinch_active: bool,
    pub volume_level_hud: Option<f32>,
    pub cursor_frozen: bool,
    pub hand: Option<HandHudOverlay>,
    pub recorder_phase: Option<RecorderPhase>,
}

pub fn run_esp_hud_loop(
    hud_rx: Receiver<HudTelemetryPacket>,
    record_space_trigger: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    debug_kinematics: bool,
    initial_width: u32,
    initial_height: u32,
) {
    let conf = Conf {
        window_title: "gesture-esp-rs".to_string(),
        window_width: initial_width as i32,
        window_height: initial_height as i32,
        high_dpi: true,
        window_resizable: true,
        sample_count: 4,
        ..Default::default()
    };

    macroquad::Window::from_config(conf, async move {
        let custom_font = load_optional_hud_font("assets/fonts/JetBrainsMono-Bold.ttf");
        let font_ref = custom_font.as_ref();

        let mut camera_tex: Option<Texture2D> = None;
        let mut rgba_staging: Vec<u8> = Vec::new();
        let mut last_seq: u64 = 0;
        let mut latest_packet: Option<HudTelemetryPacket> = None;
        let mut latency_history: VecDeque<f32> = VecDeque::with_capacity(64);

        while !shutdown.load(Ordering::Relaxed) {
            if is_key_pressed(KeyCode::Escape) || is_key_pressed(KeyCode::Q) {
                shutdown.store(true, Ordering::SeqCst);
                break;
            }

            if is_key_pressed(KeyCode::Space) {
                record_space_trigger.store(true, Ordering::SeqCst);
            }

            while let Ok(pkt) = hud_rx.try_recv() {
                latest_packet = Some(pkt);
            }

            clear_background(Color::new(0.02, 0.03, 0.05, 1.0));
            let sw = screen_width();
            let sh = screen_height();

            if let Some(ref pkt) = latest_packet {
                if pkt.frame_sequence != last_seq
                    && pkt.frame_width > 0
                    && pkt.frame_height > 0
                    && pkt.frame_rgb.len() >= (pkt.frame_width * pkt.frame_height * 3) as usize
                {
                    last_seq = pkt.frame_sequence;
                    if latency_history.len() >= 60 {
                        latency_history.pop_front();
                    }
                    latency_history.push_back(pkt.inference_ms);

                    rgb_to_rgba_buffer(&pkt.frame_rgb, &mut rgba_staging);
                    let mq_img = MqImage {
                        bytes: rgba_staging.clone(),
                        width: pkt.frame_width as u16,
                        height: pkt.frame_height as u16,
                    };

                    match camera_tex {
                        Some(ref tex)
                            if tex.width() as u32 == pkt.frame_width
                                && tex.height() as u32 == pkt.frame_height =>
                        {
                            tex.update(&mq_img);
                        }
                        _ => {
                            let tex = Texture2D::from_image(&mq_img);
                            tex.set_filter(FilterMode::Linear);
                            camera_tex = Some(tex);
                        }
                    }
                }

                let view_rect = compute_aspect_fit_rect(
                    pkt.frame_width as f32,
                    pkt.frame_height as f32,
                    sw,
                    sh,
                );
                if let Some(ref tex) = camera_tex {
                    draw_texture_ex(
                        tex,
                        view_rect.x,
                        view_rect.y,
                        WHITE,
                        DrawTextureParams {
                            dest_size: Some(macroquad::math::vec2(view_rect.w, view_rect.h)),
                            ..Default::default()
                        },
                    );
                }

                if let Some(ref hand) = pkt.hand {
                    draw_hand_esp_overlay(pkt, hand, view_rect, font_ref, debug_kinematics);
                }

                draw_system_telemetry_panel(pkt, &latency_history, font_ref);

                if let Some(ref phase) = pkt.recorder_phase {
                    draw_recorder_banner(phase, sw, sh, font_ref);
                }
            } else {
                hud_text(
                    "Starting camera...",
                    sw * 0.5 - 90.0,
                    sh * 0.5,
                    20,
                    TacticalPalette::TACTICAL_CYAN,
                    font_ref,
                );
            }

            next_frame().await;
        }

        shutdown.store(true, Ordering::SeqCst);
        let _ = camera_tex.take();
    });
}

fn draw_hand_esp_overlay(
    pkt: &HudTelemetryPacket,
    hand: &HandHudOverlay,
    view: Rect,
    font: Option<&Font>,
    debug_kinematics: bool,
) {
    let state_color = TacticalPalette::color_for_state(&pkt.fsm_state);

    let mut pts_2d = [(0.0f32, 0.0f32, 0.0f32); 21];
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for (i, lm) in hand.landmarks.iter().enumerate() {
        let sx = view.x + lm.x * view.w;
        let sy = view.y + lm.y * view.h;
        pts_2d[i] = (sx, sy, lm.z);
        min_x = min_x.min(sx);
        min_y = min_y.min(sy);
        max_x = max_x.max(sx);
        max_y = max_y.max(sy);
    }

    let pad = 24.0;
    let bx0 = (min_x - pad).max(view.x + 2.0);
    let by0 = (min_y - pad).max(view.y + 2.0);
    let bx1 = (max_x + pad).min(view.x + view.w - 2.0);
    let by1 = (max_y + pad).min(view.y + view.h - 2.0);
    let bw = (bx1 - bx0).max(20.0);
    let bh = (by1 - by0).max(20.0);

    draw_rectangle(bx0, by0, bw, bh, TacticalPalette::ESP_BOX_TINT);
    draw_corner_brackets(bx0, by0, bx1, by1, 18.0, 3.0, state_color);

    let bone_color = Color::new(state_color.r, state_color.g, state_color.b, 0.78);
    for &(a, b) in &HAND_CONNECTIONS {
        let (ax, ay, az) = pts_2d[a];
        let (bx, by, bz) = pts_2d[b];
        let mean_z = 0.5 * (az + bz);
        let thickness = (2.8 - mean_z * 8.0).clamp(1.4, 4.2);
        draw_line(ax, ay, bx, by, thickness, bone_color);
    }

    for (idx, &(jx, jy, jz)) in pts_2d.iter().enumerate() {
        let radius = (6.0 - jz * 15.0).clamp(2.5, 9.0);
        draw_circle(jx, jy, radius, state_color);
        draw_circle_lines(jx, jy, radius + 1.2, 1.0, TacticalPalette::HUD_PANEL_BG);

        if debug_kinematics {
            if let Some(finger_pos) = PIP_LANDMARK_INDICES.iter().position(|&pip| pip == idx) {
                let deg = hand.pip_angles_deg[finger_pos].round() as i32;
                hud_text(
                    &format!("{deg}°"),
                    jx + 8.0,
                    jy - 4.0,
                    14,
                    TacticalPalette::AMBER_DWELL,
                    font,
                );
            }
        }
    }

    let (t4x, t4y, _) = pts_2d[4];
    let (i8x, i8y, _) = pts_2d[8];
    let caliper_color = if pkt.pinch_active {
        TacticalPalette::NEON_GREEN
    } else {
        TacticalPalette::TACTICAL_CYAN
    };
    draw_dashed_line(t4x, t4y, i8x, i8y, 6.0, 4.0, 1.8, caliper_color);

    let mid_x = 0.5 * (t4x + i8x);
    let mid_y = 0.5 * (t4y + i8y);
    let reticle_r = (hand.pinch_index_dist * 190.0).clamp(5.0, 18.0);
    if pkt.pinch_active {
        draw_circle(mid_x, mid_y, 7.5, TacticalPalette::NEON_GREEN);
    } else {
        draw_circle_lines(mid_x, mid_y, reticle_r, 1.8, caliper_color);
    }
    hud_text(
        &format!(
            "d:{:.3} | {:.0}°",
            hand.pinch_index_dist, hand.pinch_aperture_deg
        ),
        mid_x + 11.0,
        mid_y + 4.0,
        13,
        caliper_color,
        font,
    );

    if pkt.action_label.starts_with("AIR_MOUSE") {
        let ax = 0.70 * pts_2d[5].0 + 0.30 * pts_2d[9].0;
        let ay = 0.70 * pts_2d[5].1 + 0.30 * pts_2d[9].1;
        let anchor_col = if pkt.cursor_frozen {
            TacticalPalette::AMBER_DWELL
        } else {
            TacticalPalette::NEON_GREEN
        };
        draw_circle_lines(ax, ay, 8.0, 1.6, anchor_col);
        draw_line(ax - 12.0, ay, ax + 12.0, ay, 1.4, anchor_col);
        draw_line(ax, ay - 12.0, ax, ay + 12.0, 1.4, anchor_col);
    }

    if let Some(vol) = pkt.volume_level_hud {
        let gx = (bx1 + 16.0).min(view.x + view.w - 36.0);
        let gw = 12.0;
        draw_rectangle(gx, by0, gw, bh, TacticalPalette::HUD_PANEL_BG);
        draw_rectangle_lines(gx, by0, gw, bh, 1.5, TacticalPalette::NEON_GREEN);
        let fill_h = bh * vol.clamp(0.0, 1.0);
        draw_rectangle(
            gx + 2.0,
            by1 - fill_h,
            gw - 4.0,
            fill_h,
            TacticalPalette::NEON_GREEN,
        );
        let pct = (vol * 100.0).round() as i32;
        hud_text(
            &format!("VOL: {pct}%"),
            gx - 12.0,
            by0 - 8.0,
            14,
            TacticalPalette::NEON_GREEN,
            font,
        );
    }

    let engine_src = if pkt.from_mlp { "MLP" } else { "VEC" };
    let line1 = format!(
        "[HAND: {} {:.0}%] | {}: {} ({:.1}%)",
        hand.handedness.label(),
        hand.presence_score * 100.0,
        engine_src,
        pkt.tag.as_key(),
        pkt.confidence * 100.0
    );
    let freeze_badge = if pkt.cursor_frozen { " [HOLD]" } else { "" };
    let line2 = format!(
        "STATE: {} -> {}{}",
        pkt.fsm_state.short_label(),
        pkt.action_label,
        freeze_badge
    );

    let header_h = 44.0;
    let header_w = bw.max(310.0);
    let header_y = (by0 - header_h - 6.0).max(view.y + 4.0);
    draw_rectangle(bx0, header_y, header_w, header_h, TacticalPalette::HUD_PANEL_BG);
    draw_rectangle_lines(bx0, header_y, header_w, header_h, 1.5, state_color);

    hud_text(&line1, bx0 + 8.0, header_y + 17.0, 14, state_color, font);
    hud_text(
        &line2,
        bx0 + 8.0,
        header_y + 36.0,
        13,
        TacticalPalette::TEXT_PRIMARY,
        font,
    );

    let progress = pkt.fsm_state.dwell_progress();
    if progress > 0.001 {
        let bar_y = (by1 + 6.0).min(view.y + view.h - 10.0);
        let bar_h = 6.0;
        draw_rectangle(bx0, bar_y, bw, bar_h, TacticalPalette::HUD_PANEL_BG);
        draw_rectangle(bx0, bar_y, bw * progress.clamp(0.0, 1.0), bar_h, state_color);
        draw_rectangle_lines(bx0, bar_y, bw, bar_h, 1.0, state_color);
    }
}

fn draw_system_telemetry_panel(
    pkt: &HudTelemetryPacket,
    latency_history: &VecDeque<f32>,
    font: Option<&Font>,
) {
    let px = 14.0;
    let py = 14.0;
    let pw = 355.0;
    let ph = 142.0;

    draw_rectangle(px, py, pw, ph, TacticalPalette::HUD_PANEL_BG);
    draw_rectangle_lines(px, py, pw, ph, 1.5, TacticalPalette::HUD_BORDER_DIM);

    let fps = get_fps();
    let is_locked = matches!(pkt.fsm_state, FsmState::Locked { .. });
    let clutch_str = if is_locked { "LOCKED" } else { "ARMED" };
    let clutch_col = if is_locked {
        TacticalPalette::CRIMSON_LOCK
    } else {
        TacticalPalette::NEON_GREEN
    };

    let roi_mode = match pkt.hand {
        Some(ref h) if h.used_palm_detector => "PALM",
        Some(_) => "ROI",
        None => "SCAN",
    };

    hud_text(
        &format!(
            "FPS: {:>3} | INF: {:>4.1}ms | E2E: {:>4.1}ms | {}",
            fps, pkt.inference_ms, pkt.pipeline_e2e_ms, roi_mode
        ),
        px + 10.0,
        py + 22.0,
        13,
        TacticalPalette::TACTICAL_CYAN,
        font,
    );
    hud_text(
        &format!("EP : {}", pkt.execution_provider),
        px + 10.0,
        py + 42.0,
        13,
        TacticalPalette::TEXT_PRIMARY,
        font,
    );
    hud_text(
        &format!("CLUTCH: [{clutch_str}]"),
        px + 10.0,
        py + 62.0,
        14,
        clutch_col,
        font,
    );
    hud_text(
        &format!("CFG: {}", pkt.config_status),
        px + 10.0,
        py + 82.0,
        13,
        TacticalPalette::TEXT_DIM,
        font,
    );

    let gx = px + 10.0;
    let gy = py + 94.0;
    let gw = pw - 20.0;
    let gh = 36.0;
    draw_rectangle(gx, gy, gw, gh, Color::new(0.01, 0.02, 0.03, 0.85));
    draw_rectangle_lines(gx, gy, gw, gh, 1.0, TacticalPalette::HUD_BORDER_DIM);

    if latency_history.len() >= 2 {
        let max_ms = latency_history
            .iter()
            .copied()
            .fold(16.0f32, f32::max);
        let step_x = gw / ((latency_history.len() - 1).max(1) as f32);
        for i in 1..latency_history.len() {
            let x0 = gx + ((i - 1) as f32) * step_x;
            let x1 = gx + (i as f32) * step_x;
            let y0 = gy + gh - (latency_history[i - 1] / max_ms).clamp(0.0, 1.0) * (gh - 4.0) - 2.0;
            let y1 = gy + gh - (latency_history[i] / max_ms).clamp(0.0, 1.0) * (gh - 4.0) - 2.0;
            draw_line(x0, y0, x1, y1, 1.6, TacticalPalette::NEON_GREEN);
        }
    }
}

fn draw_recorder_banner(
    phase: &RecorderPhase,
    sw: f32,
    sh: f32,
    font: Option<&Font>,
) {
    let bw = 560.0f32.min(sw - 32.0);
    let bh = 58.0;
    let bx = 0.5 * (sw - bw);
    let by = sh - bh - 18.0;

    draw_rectangle(bx, by, bw, bh, TacticalPalette::HUD_PANEL_BG);

    match phase {
        RecorderPhase::WaitingForTrigger { target_label } => {
            draw_rectangle_lines(bx, by, bw, bh, 2.0, TacticalPalette::AMBER_DWELL);
            hud_text(
                &format!("RECORDER [{}] - PRESS SPACE TO CAPTURE", target_label),
                bx + 14.0,
                by + 34.0,
                15,
                TacticalPalette::AMBER_DWELL,
                font,
            );
        }
        RecorderPhase::Capturing {
            target_label,
            captured,
            total,
        } => {
            draw_rectangle_lines(bx, by, bw, bh, 2.0, TacticalPalette::NEON_GREEN);
            let frac = (*captured as f32) / ((*total).max(1) as f32);
            hud_text(
                &format!(
                    "RECORDING [{}]: {}/{} ({:.0}%)",
                    target_label,
                    captured,
                    total,
                    frac * 100.0
                ),
                bx + 14.0,
                by + 25.0,
                15,
                TacticalPalette::NEON_GREEN,
                font,
            );
            draw_rectangle(
                bx + 14.0,
                by + 36.0,
                (bw - 28.0) * frac.clamp(0.0, 1.0),
                10.0,
                TacticalPalette::NEON_GREEN,
            );
        }
        RecorderPhase::Completed {
            target_label,
            accuracy,
        } => {
            draw_rectangle_lines(bx, by, bw, bh, 2.0, TacticalPalette::TACTICAL_CYAN);
            hud_text(
                &format!(
                    "SAVED [{}] - ACCURACY: {:.1}% (SPACE TO RECORD AGAIN)",
                    target_label,
                    (*accuracy as f32) * 0.1
                ),
                bx + 14.0,
                by + 34.0,
                14,
                TacticalPalette::TACTICAL_CYAN,
                font,
            );
        }
    }
}

fn draw_corner_brackets(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    arm: f32,
    thickness: f32,
    color: Color,
) {
    let lx = arm.min(0.45 * (x1 - x0).abs());
    let ly = arm.min(0.45 * (y1 - y0).abs());

    draw_line(x0, y0, x0 + lx, y0, thickness, color);
    draw_line(x0, y0, x0, y0 + ly, thickness, color);
    draw_line(x1, y0, x1 - lx, y0, thickness, color);
    draw_line(x1, y0, x1, y0 + ly, thickness, color);
    draw_line(x0, y1, x0 + lx, y1, thickness, color);
    draw_line(x0, y1, x0, y1 - ly, thickness, color);
    draw_line(x1, y1, x1 - lx, y1, thickness, color);
    draw_line(x1, y1, x1, y1 - ly, thickness, color);
}

#[allow(clippy::too_many_arguments)]
fn draw_dashed_line(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    dash_len: f32,
    gap_len: f32,
    thickness: f32,
    color: Color,
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist < 1.0 {
        return;
    }
    let ux = dx / dist;
    let uy = dy / dist;
    let stride = (dash_len + gap_len).max(2.0);
    let mut t = 0.0f32;
    while t < dist {
        let t_end = (t + dash_len).min(dist);
        draw_line(
            x0 + ux * t,
            y0 + uy * t,
            x0 + ux * t_end,
            y0 + uy * t_end,
            thickness,
            color,
        );
        t += stride;
    }
}

fn hud_text(text: &str, x: f32, y: f32, font_size: u16, color: Color, font: Option<&Font>) {
    draw_text_ex(
        text,
        x,
        y,
        TextParams {
            font,
            font_size,
            font_scale: 1.0,
            color,
            ..Default::default()
        },
    );
}

fn load_optional_hud_font(path: impl AsRef<Path>) -> Option<Font> {
    let bytes = fs::read(path.as_ref()).ok()?;
    load_ttf_font_from_bytes(&bytes).ok()
}

fn compute_aspect_fit_rect(src_w: f32, src_h: f32, dst_w: f32, dst_h: f32) -> Rect {
    let scale = (dst_w / src_w.max(1.0)).min(dst_h / src_h.max(1.0));
    let w = src_w * scale;
    let h = src_h * scale;
    Rect::new(0.5 * (dst_w - w), 0.5 * (dst_h - h), w, h)
}

#[inline]
fn rgb_to_rgba_buffer(rgb: &[u8], rgba_out: &mut Vec<u8>) {
    let pixel_count = rgb.len() / 3;
    let target_len = pixel_count * 4;
    if rgba_out.len() != target_len {
        rgba_out.resize(target_len, 255);
    }
    for (src, dst) in rgb.chunks_exact(3).zip(rgba_out.chunks_exact_mut(4)) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = 255;
    }
}
