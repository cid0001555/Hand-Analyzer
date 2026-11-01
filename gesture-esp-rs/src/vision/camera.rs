use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
};
use nokhwa::Camera;
use parking_lot::RwLock;
use tracing::{error, info, warn};

use crate::config::AppConfig;

#[derive(Debug, Clone)]
pub struct RawFrame {
    pub rgb: Arc<Vec<u8>>,
    pub width: u32,
    pub height: u32,
    pub captured_at: Instant,
    pub sequence: u64,
}

pub fn spawn_camera_thread(
    config: Arc<RwLock<AppConfig>>,
    frame_tx: Sender<RawFrame>,
    frame_drain_rx: Receiver<RawFrame>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("camera-worker".to_string())
        .spawn(move || {
            let init_cfg = config.read().general.clone();
            let mut camera = match open_camera_with_fallbacks(
                init_cfg.camera_index,
                init_cfg.capture_width,
                init_cfg.capture_height,
                init_cfg.target_fps,
            ) {
                Ok(cam) => cam,
                Err(err) => {
                    error!(error = %err, "Failed to initialize camera");
                    return;
                }
            };

            if let Err(err) = camera.open_stream() {
                error!(error = %err, "Failed to open camera stream");
                return;
            }

            let actual_fmt = camera.camera_format();
            info!(
                index = init_cfg.camera_index,
                width = actual_fmt.width(),
                height = actual_fmt.height(),
                fps = actual_fmt.frame_rate(),
                fourcc = ?actual_fmt.format(),
                "Camera stream started"
            );

            let mut sequence: u64 = 0;

            while !shutdown.load(Ordering::Relaxed) {
                match camera.frame() {
                    Ok(buffer) => {
                        let captured_at = Instant::now();
                        match buffer.decode_image::<RgbFormat>() {
                            Ok(decoded) => {
                                let width = decoded.width();
                                let height = decoded.height();
                                let mut raw_bytes = decoded.into_raw();

                                let mirror = config.read().general.mirror_horizontal;
                                if mirror {
                                    flip_rgb_horizontal_in_place(&mut raw_bytes, width as usize, height as usize);
                                }

                                sequence = sequence.wrapping_add(1);
                                let packet = RawFrame {
                                    rgb: Arc::new(raw_bytes),
                                    width,
                                    height,
                                    captured_at,
                                    sequence,
                                };

                                while frame_drain_rx.try_recv().is_ok() {}
                                let _ = frame_tx.try_send(packet);
                            }
                            Err(err) => {
                                warn!(error = %err, "Frame decode error");
                            }
                        }
                    }
                    Err(err) => {
                        warn!(error = %err, "Camera capture error");
                        thread::sleep(Duration::from_millis(8));
                    }
                }
            }

            let _ = camera.stop_stream();
        })
        .expect("Failed to spawn camera thread")
}

fn open_camera_with_fallbacks(
    index: u32,
    width: u32,
    height: u32,
    fps: u32,
) -> Result<Camera, String> {
    let cam_idx = CameraIndex::Index(index);
    let res = Resolution::new(width, height);

    let mjpeg_req = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
        CameraFormat::new(res, FrameFormat::MJPEG, fps),
    ));
    if let Ok(cam) = Camera::new(cam_idx.clone(), mjpeg_req) {
        return Ok(cam);
    }

    let yuyv_req = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
        CameraFormat::new(res, FrameFormat::YUYV, fps),
    ));
    if let Ok(cam) = Camera::new(cam_idx.clone(), yuyv_req) {
        return Ok(cam);
    }

    let fallback_req =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    Camera::new(cam_idx, fallback_req).map_err(|e| e.to_string())
}

#[inline]
pub fn flip_rgb_horizontal_in_place(rgb: &mut [u8], width: usize, height: usize) {
    let row_stride = width * 3;
    if rgb.len() < row_stride * height || width <= 1 {
        return;
    }
    let half_w = width / 2;
    for row in rgb.chunks_exact_mut(row_stride) {
        for x in 0..half_w {
            let left = x * 3;
            let right = (width - 1 - x) * 3;
            row.swap(left, right);
            row.swap(left + 1, right + 1);
            row.swap(left + 2, right + 2);
        }
    }
}
