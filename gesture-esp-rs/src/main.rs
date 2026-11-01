use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use gesture_esp_rs::actions::spawn_action_dispatcher_thread;
use gesture_esp_rs::classifier::{
    recorder::train_mlp_from_csv, GestureTag, HybridClassifier, PoseDatasetRecorder,
};
use gesture_esp_rs::config::LiveConfigHandle;
use gesture_esp_rs::fsm::{FrameObservation, OsActionEvent, StateMachine};
use gesture_esp_rs::math::{HandKinematics, HandNormalizer, SkeletonFilterBank};
use gesture_esp_rs::ui::{run_esp_hud_loop, HandHudOverlay, HudTelemetryPacket};
use gesture_esp_rs::vision::{spawn_camera_thread, OnnxVisionEngine, RawFrame};

#[derive(Debug, Parser)]
#[command(name = "gesture-esp-rs", version)]
struct CliArgs {
    #[arg(short, long, default_value = "config/mappings.toml")]
    config: PathBuf,

    #[arg(long)]
    headless: bool,

    #[arg(long, value_name = "GESTURE_NAME")]
    record: Option<String>,

    #[arg(long)]
    train_only: bool,

    #[arg(long)]
    debug_kinematics: bool,

    #[arg(long)]
    dry_run: bool,

    #[arg(long, default_value = "assets/models/palm_detection_full.onnx")]
    palm_model: PathBuf,

    #[arg(long, default_value = "assets/models/hand_landmark_full.onnx")]
    landmark_model: PathBuf,

    #[arg(long, default_value = "assets/models/custom_gestures_mlp.bin")]
    mlp_model: PathBuf,

    #[arg(long, default_value = "data/recorded_poses.csv")]
    dataset_csv: PathBuf,
}

fn main() -> Result<()> {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .init();

    let args = CliArgs::parse();

    if args.train_only {
        info!(csv = %args.dataset_csv.display(), "Training MLP from CSV");
        let (_model, acc) = train_mlp_from_csv(&args.dataset_csv, &args.mlp_model)
            .context("MLP training failed")?;
        info!(
            accuracy_pct = acc * 100.0,
            output = %args.mlp_model.display(),
            "Saved MLP weights"
        );
        return Ok(());
    }

    let live_config = LiveConfigHandle::initialize(&args.config)
        .with_context(|| format!("Failed to load config from {}", args.config.display()))?;
    let initial_snapshot = live_config.snapshot();

    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown_sig = Arc::clone(&shutdown);
        ctrlc::set_handler(move || {
            info!("Shutting down");
            shutdown_sig.store(true, Ordering::SeqCst);
        })
        .context("Failed to install Ctrl+C handler")?;
    }

    let (frame_tx, frame_rx) = bounded::<RawFrame>(1);
    let frame_drain_rx = frame_rx.clone();

    let (action_tx, action_rx) = bounded::<OsActionEvent>(16);

    let (hud_tx, hud_rx) = bounded::<HudTelemetryPacket>(1);
    let hud_drain_rx = hud_rx.clone();

    let record_space_trigger = Arc::new(AtomicBool::new(false));

    let cam_handle = spawn_camera_thread(
        live_config.shared_config(),
        frame_tx,
        frame_drain_rx,
        Arc::clone(&shutdown),
    );

    let effective_dry_run = args.dry_run || args.record.is_some();
    let dispatcher_handle =
        spawn_action_dispatcher_thread(action_rx, Arc::clone(&shutdown), effective_dry_run);

    let vision_handle = spawn_vision_fsm_thread(VisionWorkerParams {
        live_config: &live_config,
        frame_rx,
        action_tx,
        hud_tx: if args.headless { None } else { Some((hud_tx, hud_drain_rx)) },
        palm_model: args.palm_model.clone(),
        landmark_model: args.landmark_model.clone(),
        mlp_model: args.mlp_model.clone(),
        dataset_csv: args.dataset_csv.clone(),
        record_gesture: args.record.clone(),
        headless: args.headless,
        record_space_trigger: Arc::clone(&record_space_trigger),
        shutdown: Arc::clone(&shutdown),
    })?;

    if args.headless {
        info!("Running in headless mode");
        let mut known_gen = live_config.generation();
        while !shutdown.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(100));
            let cur_gen = live_config.generation();
            if cur_gen != known_gen {
                known_gen = cur_gen;
                let tel = live_config.telemetry();
                info!(status = %tel.status_text, "Config updated");
            }
        }
    } else {
        run_esp_hud_loop(
            hud_rx,
            record_space_trigger,
            Arc::clone(&shutdown),
            args.debug_kinematics,
            initial_snapshot.general.capture_width,
            initial_snapshot.general.capture_height,
        );
    }

    shutdown.store(true, Ordering::SeqCst);
    let _ = cam_handle.join();
    let _ = vision_handle.join();
    let _ = dispatcher_handle.join();
    info!("Stopped");
    Ok(())
}

struct VisionWorkerParams<'a> {
    live_config: &'a LiveConfigHandle,
    frame_rx: Receiver<RawFrame>,
    action_tx: Sender<OsActionEvent>,
    hud_tx: Option<(Sender<HudTelemetryPacket>, Receiver<HudTelemetryPacket>)>,
    palm_model: PathBuf,
    landmark_model: PathBuf,
    mlp_model: PathBuf,
    dataset_csv: PathBuf,
    record_gesture: Option<String>,
    headless: bool,
    record_space_trigger: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
}

fn spawn_vision_fsm_thread(params: VisionWorkerParams<'_>) -> Result<JoinHandle<()>> {
    let shared_cfg = params.live_config.shared_config();
    let shared_tel = params.live_config.shared_telemetry();
    let shared_gen = params.live_config.shared_generation();

    let init_cfg = shared_cfg.read().clone();
    let mut engine = OnnxVisionEngine::new(
        &params.palm_model,
        &params.landmark_model,
        &init_cfg.general.execution_provider,
    )?;

    let frame_rx = params.frame_rx;
    let action_tx = params.action_tx;
    let hud_channel = params.hud_tx;
    let mlp_model_path = params.mlp_model;
    let dataset_csv_path = params.dataset_csv;
    let record_gesture = params.record_gesture;
    let headless = params.headless;
    let record_space_trigger = params.record_space_trigger;
    let shutdown = params.shutdown;

    let handle = thread::Builder::new()
        .name("vision-worker".to_string())
        .spawn(move || {
            let mut active_cfg = shared_cfg.read().clone();
            let mut active_gen = shared_gen.load(Ordering::Relaxed);

            let mut screen_filters = SkeletonFilterBank::from_config(active_cfg.filter);
            let mut pinch_fast_filters = SkeletonFilterBank::new(4.2, 0.25, 1.0);
            let mut world_filters = SkeletonFilterBank::from_config(active_cfg.filter);
            let mut classifier = HybridClassifier::new(Some(&mlp_model_path));
            let mut fsm = StateMachine::new(active_cfg.fsm.require_clutch);

            let mut recorder = record_gesture.as_deref().map(|label| {
                PoseDatasetRecorder::new(label, &dataset_csv_path, &mlp_model_path)
            });

            let headless_record_autostart = if headless && recorder.is_some() {
                info!("Recording starts in 3 seconds");
                Some(Instant::now() + Duration::from_secs(3))
            } else {
                None
            };
            let mut headless_triggered = false;
            let mut last_frame_instant: Option<Instant> = None;

            while !shutdown.load(Ordering::Relaxed) {
                let frame = match frame_rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(f) => f,
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                };

                let latest_gen = shared_gen.load(Ordering::Relaxed);
                if latest_gen != active_gen {
                    active_gen = latest_gen;
                    active_cfg = shared_cfg.read().clone();
                    screen_filters.update_params(active_cfg.filter);
                    world_filters.update_params(active_cfg.filter);
                }

                if let Some(ref mut rec) = recorder {
                    if record_space_trigger.swap(false, Ordering::SeqCst) {
                        rec.trigger_capture();
                    }
                    if let Some(auto_at) = headless_record_autostart {
                        if !headless_triggered && Instant::now() >= auto_at {
                            headless_triggered = true;
                            rec.trigger_capture();
                        }
                    }
                }

                let now = Instant::now();
                let dt_sec = last_frame_instant
                    .map(|prev| now.saturating_duration_since(prev).as_secs_f32())
                    .unwrap_or(1.0 / 60.0);
                last_frame_instant = Some(now);

                let inf_start = Instant::now();
                let infer_res = engine.infer_hand(
                    &frame.rgb,
                    frame.width,
                    frame.height,
                    active_cfg.general.mirror_horizontal,
                );
                let inference_ms = inf_start.elapsed().as_secs_f32() * 1000.0;

                match infer_res {
                    Ok(Some(raw_hand)) => {
                        let filtered_screen =
                            screen_filters.filter_skeleton(&raw_hand.screen_landmarks, dt_sec);
                        let fast_pinch_screen =
                            pinch_fast_filters.filter_skeleton(&raw_hand.screen_landmarks, dt_sec);
                        let filtered_world =
                            world_filters.filter_skeleton(&raw_hand.world_landmarks, dt_sec);

                        let aspect = (frame.width.max(1) as f32) / (frame.height.max(1) as f32);
                        let mut iso_screen = fast_pinch_screen;
                        for pt in &mut iso_screen {
                            pt.x *= aspect;
                        }

                        let canonical = HandNormalizer::normalize(&iso_screen);
                        let kinematics = HandKinematics::compute(&filtered_world, &canonical);

                        if let Some(ref mut rec) = recorder {
                            match rec.push_canonical_sample(&canonical.feature_vector) {
                                Ok(Some(new_mlp)) => {
                                    classifier.set_mlp(new_mlp);
                                    if headless {
                                        info!("Recording complete");
                                        shutdown.store(true, Ordering::SeqCst);
                                    }
                                }
                                Ok(None) => {}
                                Err(err) => {
                                    warn!(error = %err, "Recorder error");
                                }
                            }
                        }

                        let cls = classifier.classify(&canonical, &kinematics);
                        let obs = FrameObservation {
                            now,
                            dt_sec,
                            tag: cls.tag.clone(),
                            confidence: cls.confidence,
                            landmarks: filtered_screen,
                            pinch_index_dist: kinematics.pinch_index_dist,
                            pinch_aperture_deg: kinematics.pinch_aperture_deg,
                            pinch_middle_dist: kinematics.pinch_middle_dist,
                            middle_aperture_deg: kinematics.middle_aperture_deg,
                            middle_pip_deg: kinematics.fingers[2].pip_deg,
                        };

                        let fsm_out = fsm.update(&obs, &active_cfg);

                        for ev in fsm_out.events {
                            let _ = action_tx.try_send(ev);
                        }

                        if let Some((ref tx, ref drain_rx)) = hud_channel {
                            let pip_angles_deg = [
                                kinematics.fingers[0].pip_deg,
                                kinematics.fingers[1].pip_deg,
                                kinematics.fingers[2].pip_deg,
                                kinematics.fingers[3].pip_deg,
                                kinematics.fingers[4].pip_deg,
                            ];

                            let pkt = HudTelemetryPacket {
                                frame_rgb: Arc::clone(&frame.rgb),
                                frame_width: frame.width,
                                frame_height: frame.height,
                                frame_sequence: frame.sequence,
                                inference_ms,
                                pipeline_e2e_ms: frame.captured_at.elapsed().as_secs_f32() * 1000.0,
                                execution_provider: engine.active_provider().to_string(),
                                config_status: shared_tel.read().status_text.clone(),
                                tag: cls.tag,
                                confidence: cls.confidence,
                                from_mlp: cls.from_mlp,
                                fsm_state: fsm_out.state,
                                action_label: fsm_out.active_action_label,
                                pinch_active: fsm_out.pinch_active,
                                volume_level_hud: fsm_out.volume_level_hud,
                                cursor_frozen: fsm_out.cursor_frozen,
                                hand: Some(HandHudOverlay {
                                    landmarks: filtered_screen,
                                    handedness: raw_hand.handedness,
                                    presence_score: raw_hand.presence_score,
                                    pip_angles_deg,
                                    pinch_index_dist: kinematics.pinch_index_dist,
                                    pinch_aperture_deg: kinematics.pinch_aperture_deg,
                                    used_palm_detector: raw_hand.used_palm_detector,
                                }),
                                recorder_phase: recorder.as_ref().map(|r| r.phase()),
                            };
                            while drain_rx.try_recv().is_ok() {}
                            let _ = tx.try_send(pkt);
                        }
                    }
                    Ok(None) => {
                        screen_filters.reset();
                        pinch_fast_filters.reset();
                        world_filters.reset();
                        let fsm_out = fsm.on_hand_lost(now);
                        for ev in fsm_out.events {
                            let _ = action_tx.try_send(ev);
                        }

                        if let Some((ref tx, ref drain_rx)) = hud_channel {
                            let pkt = HudTelemetryPacket {
                                frame_rgb: Arc::clone(&frame.rgb),
                                frame_width: frame.width,
                                frame_height: frame.height,
                                frame_sequence: frame.sequence,
                                inference_ms,
                                pipeline_e2e_ms: frame.captured_at.elapsed().as_secs_f32() * 1000.0,
                                execution_provider: engine.active_provider().to_string(),
                                config_status: shared_tel.read().status_text.clone(),
                                tag: GestureTag::Neutral,
                                confidence: 0.0,
                                from_mlp: false,
                                fsm_state: fsm_out.state,
                                action_label: fsm_out.active_action_label,
                                pinch_active: false,
                                volume_level_hud: None,
                                cursor_frozen: false,
                                hand: None,
                                recorder_phase: recorder.as_ref().map(|r| r.phase()),
                            };
                            while drain_rx.try_recv().is_ok() {}
                            let _ = tx.try_send(pkt);
                        }
                    }
                    Err(err) => {
                        error!(error = %err, "Inference error");
                        engine.reset_roi_cache();
                    }
                }
            }
        })
        .context("Failed to spawn vision thread")?;

    Ok(handle)
}
