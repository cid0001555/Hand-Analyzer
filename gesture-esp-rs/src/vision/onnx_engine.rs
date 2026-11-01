use std::f32::consts::FRAC_PI_2;
use std::path::Path;

use glam::Vec3A;
use ort::ep::{CPU, ExecutionProviderDispatch};
#[cfg(target_os = "windows")]
use ort::ep::DirectML;
#[cfg(feature = "cuda")]
use ort::ep::{CUDA, TensorRT};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Tensor, ValueType};
use tracing::info;

use crate::error::{DomainResult, GestureEspError};
use crate::vision::preprocessor::{
    extract_oriented_roi_tensor, letterbox_rgb_to_tensor, LetterboxMeta, TensorLayout,
};
use crate::vision::skeleton::{normalize_radians, HandSkeleton, Handedness, OrientedRoi};

const PALM_SCORE_THRESHOLD: f32 = 0.60;
const PALM_NMS_IOU_THRESHOLD: f32 = 0.30;
const LANDMARK_PRESENCE_THRESHOLD: f32 = 0.50;
const PALM_ENLARGE_SCALE: f32 = 2.6;
const PALM_SHIFT_ALONG_HAND: f32 = 0.5;

#[derive(Debug, Clone, Copy)]
struct SsdAnchor {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone)]
struct PalmDetection {
    score: f32,
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    keypoints: [(f32, f32); 7],
}

pub struct OnnxVisionEngine {
    palm_session: Session,
    palm_input_name: String,
    palm_layout: TensorLayout,
    palm_input_size: usize,
    anchors_192: Vec<SsdAnchor>,
    anchors_128: Vec<SsdAnchor>,

    landmark_session: Session,
    landmark_input_name: String,
    landmark_layout: TensorLayout,
    landmark_input_size: usize,

    active_provider_label: String,
    cached_roi: Option<OrientedRoi>,
}

impl OnnxVisionEngine {
    pub fn new(
        palm_model_path: impl AsRef<Path>,
        landmark_model_path: impl AsRef<Path>,
        provider_preference: &str,
    ) -> DomainResult<Self> {
        let palm_path = palm_model_path.as_ref();
        let landmark_path = landmark_model_path.as_ref();

        if !palm_path.exists() {
            return Err(GestureEspError::ModelNotFound(palm_path.to_path_buf()));
        }
        if !landmark_path.exists() {
            return Err(GestureEspError::ModelNotFound(landmark_path.to_path_buf()));
        }

        let (providers, active_provider_label) = build_execution_providers(provider_preference);
        let env = ort::init().with_name("gesture-esp-rs").build()?;

        let palm_session = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort::Error::from)?
            .with_intra_threads(2)
            .map_err(ort::Error::from)?
            .with_execution_providers(providers.clone())
            .map_err(ort::Error::from)?
            .commit_from_file(palm_path)?;

        let landmark_session = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort::Error::from)?
            .with_intra_threads(2)
            .map_err(ort::Error::from)?
            .with_execution_providers(providers)
            .map_err(ort::Error::from)?
            .commit_from_file(landmark_path)?;

        let (palm_input_name, palm_layout, palm_input_size) =
            inspect_model_input(&palm_session, 192);
        let (landmark_input_name, landmark_layout, landmark_input_size) =
            inspect_model_input(&landmark_session, 224);

        info!(
            provider = %active_provider_label,
            palm_layout = ?palm_layout,
            palm_size = palm_input_size,
            landmark_layout = ?landmark_layout,
            landmark_size = landmark_input_size,
            "ONNX sessions initialized"
        );

        Ok(Self {
            palm_session,
            palm_input_name,
            palm_layout,
            palm_input_size,
            anchors_192: generate_palm_ssd_anchors(192),
            anchors_128: generate_palm_ssd_anchors(128),
            landmark_session,
            landmark_input_name,
            landmark_layout,
            landmark_input_size,
            active_provider_label,
            cached_roi: None,
        })
    }

    #[inline]
    pub fn active_provider(&self) -> &str {
        &self.active_provider_label
    }

    pub fn reset_roi_cache(&mut self) {
        self.cached_roi = None;
    }

    pub fn infer_hand(
        &mut self,
        rgb: &[u8],
        frame_width: u32,
        frame_height: u32,
        mirrored: bool,
    ) -> DomainResult<Option<HandSkeleton>> {
        if let Some(roi) = self.cached_roi {
            if let Some(mut skeleton) =
                self.run_landmark_stage(rgb, frame_width, frame_height, &roi, mirrored, false)?
            {
                let next_roi = OrientedRoi::from_landmarks(
                    &skeleton.screen_landmarks,
                    frame_width,
                    frame_height,
                );
                self.cached_roi = Some(next_roi);
                skeleton.roi = next_roi;
                return Ok(Some(skeleton));
            }
            self.cached_roi = None;
        }

        let Some(detected_roi) = self.run_palm_detector(rgb, frame_width, frame_height)? else {
            return Ok(None);
        };

        if let Some(mut skeleton) = self.run_landmark_stage(
            rgb,
            frame_width,
            frame_height,
            &detected_roi,
            mirrored,
            true,
        )? {
            let refined_roi =
                OrientedRoi::from_landmarks(&skeleton.screen_landmarks, frame_width, frame_height);
            self.cached_roi = Some(refined_roi);
            skeleton.roi = refined_roi;
            Ok(Some(skeleton))
        } else {
            self.cached_roi = None;
            Ok(None)
        }
    }

    fn run_palm_detector(
        &mut self,
        rgb: &[u8],
        frame_width: u32,
        frame_height: u32,
    ) -> DomainResult<Option<OrientedRoi>> {
        let size = self.palm_input_size;
        let (tensor_buf, letterbox) =
            letterbox_rgb_to_tensor(rgb, frame_width, frame_height, size, self.palm_layout);

        let shape = match self.palm_layout {
            TensorLayout::Nchw => [1usize, 3, size, size],
            TensorLayout::Nhwc => [1usize, size, size, 3],
        };
        let input_tensor = Tensor::from_array((shape, tensor_buf.into_boxed_slice()))?;

        let outputs = self
            .palm_session
            .run(ort::inputs![self.palm_input_name.as_str() => input_tensor])?;

        let mut regressors_buf: Option<Vec<f32>> = None;
        let mut scores_buf: Option<Vec<f32>> = None;

        for (_name, val) in outputs.iter() {
            if let Ok((out_shape, data)) = val.try_extract_tensor::<f32>() {
                let last_dim = out_shape.last().copied().unwrap_or(0) as usize;
                if last_dim == 18 {
                    regressors_buf = Some(data.to_vec());
                } else if last_dim == 1 && data.len() >= 896 {
                    scores_buf = Some(data.to_vec());
                }
            }
        }

        let (Some(regressors), Some(scores)) = (regressors_buf.as_deref(), scores_buf.as_deref()) else {
            return Err(GestureEspError::TensorShape {
                model: "palm_detection_full.onnx",
                expected: "[1, 2016, 18] and [1, 2016, 1]",
                actual: vec![outputs.len()],
            });
        };

        let num_anchors = scores.len();
        let anchors: &[SsdAnchor] = if num_anchors == self.anchors_192.len() {
            &self.anchors_192
        } else if num_anchors == self.anchors_128.len() {
            &self.anchors_128
        } else {
            return Err(GestureEspError::TensorShape {
                model: "palm_detection_full.onnx",
                expected: "2016 or 896 anchors",
                actual: vec![num_anchors],
            });
        };

        let inv_size = 1.0 / (size as f32);
        let mut candidates = Vec::new();

        for (i, anchor) in anchors.iter().enumerate() {
            let raw_logit = scores[i].clamp(-80.0, 80.0);
            let score = 1.0 / (1.0 + (-raw_logit).exp());
            if score < PALM_SCORE_THRESHOLD {
                continue;
            }

            let base = i * 18;
            if base + 18 > regressors.len() {
                break;
            }
            let cx = anchor.x + regressors[base] * inv_size;
            let cy = anchor.y + regressors[base + 1] * inv_size;
            let w = regressors[base + 2] * inv_size;
            let h = regressors[base + 3] * inv_size;

            let mut keypoints = [(0.0f32, 0.0f32); 7];
            for k in 0..7 {
                let kx = anchor.x + regressors[base + 4 + k * 2] * inv_size;
                let ky = anchor.y + regressors[base + 5 + k * 2] * inv_size;
                keypoints[k] = (kx, ky);
            }

            candidates.push(PalmDetection {
                score,
                xmin: cx - 0.5 * w,
                ymin: cy - 0.5 * h,
                xmax: cx + 0.5 * w,
                ymax: cy + 0.5 * h,
                keypoints,
            });
        }

        if candidates.is_empty() {
            return Ok(None);
        }

        let kept = non_maximum_suppression(candidates, PALM_NMS_IOU_THRESHOLD);
        let Some(best) = kept.first() else {
            return Ok(None);
        };

        Ok(Some(palm_detection_to_oriented_roi(best, &letterbox)))
    }

    fn run_landmark_stage(
        &mut self,
        rgb: &[u8],
        frame_width: u32,
        frame_height: u32,
        roi: &OrientedRoi,
        mirrored: bool,
        used_palm_detector: bool,
    ) -> DomainResult<Option<HandSkeleton>> {
        let size = self.landmark_input_size;
        let (tensor_buf, inv_affine) = extract_oriented_roi_tensor(
            rgb,
            frame_width,
            frame_height,
            roi,
            size,
            self.landmark_layout,
        );

        let shape = match self.landmark_layout {
            TensorLayout::Nchw => [1usize, 3, size, size],
            TensorLayout::Nhwc => [1usize, size, size, 3],
        };
        let input_tensor = Tensor::from_array((shape, tensor_buf.into_boxed_slice()))?;

        let outputs = self
            .landmark_session
            .run(ort::inputs![self.landmark_input_name.as_str() => input_tensor])?;

        let mut tensor_63_list: Vec<Vec<f32>> = Vec::with_capacity(2);
        let mut scalar_list: Vec<(String, f32)> = Vec::with_capacity(3);

        for (name, val) in outputs.iter() {
            if let Ok((_out_shape, data)) = val.try_extract_tensor::<f32>() {
                if data.len() == 63 {
                    tensor_63_list.push(data.to_vec());
                } else if data.len() == 1 {
                    scalar_list.push((name.to_lowercase(), data[0]));
                }
            }
        }

        if tensor_63_list.is_empty() {
            return Err(GestureEspError::TensorShape {
                model: "hand_landmark_full.onnx",
                expected: "[1, 63]",
                actual: vec![outputs.len()],
            });
        }

        let (screen_raw, world_raw) = if tensor_63_list.len() >= 2 {
            let max0 = max_abs(&tensor_63_list[0]);
            let max1 = max_abs(&tensor_63_list[1]);
            if max0 >= max1 {
                (&tensor_63_list[0], Some(&tensor_63_list[1]))
            } else {
                (&tensor_63_list[1], Some(&tensor_63_list[0]))
            }
        } else {
            (&tensor_63_list[0], None)
        };

        let (presence_score, handedness_score) = parse_landmark_scalars(&scalar_list);
        if presence_score < LANDMARK_PRESENCE_THRESHOLD {
            return Ok(None);
        }

        let inv_crop = if max_abs(screen_raw) > 2.0 {
            1.0 / (size as f32)
        } else {
            1.0
        };

        let roi_to_height = roi.size_px / (frame_height.max(1) as f32);
        let aspect = (frame_width.max(1) as f32) / (frame_height.max(1) as f32);
        let mut screen_landmarks = [Vec3A::ZERO; 21];
        for i in 0..21 {
            let base = i * 3;
            let crop_u = screen_raw[base] * inv_crop;
            let crop_v = screen_raw[base + 1] * inv_crop;
            let crop_z = screen_raw[base + 2] * inv_crop;

            let (fx, fy) = inv_affine.map_crop_norm_to_frame_norm(crop_u, crop_v);
            screen_landmarks[i] = Vec3A::new(fx, fy, crop_z * roi_to_height);
        }

        let mut world_landmarks = [Vec3A::ZERO; 21];
        if let Some(w_raw) = world_raw {
            let cos_r = roi.rotation_rad.cos();
            let sin_r = roi.rotation_rad.sin();
            for i in 0..21 {
                let base = i * 3;
                let wx = w_raw[base];
                let wy = w_raw[base + 1];
                let wz = w_raw[base + 2];
                let rx = wx * cos_r - wy * sin_r;
                let ry = wx * sin_r + wy * cos_r;
                world_landmarks[i] = Vec3A::new(rx, ry, wz);
            }
        } else {
            for i in 0..21 {
                let s = screen_landmarks[i];
                world_landmarks[i] = Vec3A::new(s.x * aspect, s.y, s.z);
            }
        }

        Ok(Some(HandSkeleton {
            screen_landmarks,
            world_landmarks,
            presence_score,
            handedness: Handedness::from_score(handedness_score, mirrored),
            roi: *roi,
            used_palm_detector,
        }))
    }
}

fn build_execution_providers(pref: &str) -> (Vec<ExecutionProviderDispatch>, String) {
    let normalized = pref.trim().to_ascii_lowercase();
    match normalized.as_str() {
        #[cfg(feature = "cuda")]
        "tensorrt" => (
            vec![
                TensorRT::default().build(),
                CUDA::default().build(),
                CPU::default().build(),
            ],
            "TensorRT -> CUDA -> CPU".to_string(),
        ),
        #[cfg(feature = "cuda")]
        "cuda" => (
            vec![CUDA::default().build(), CPU::default().build()],
            "CUDA -> CPU".to_string(),
        ),
        #[cfg(target_os = "windows")]
        "directml" | "dml" => (
            vec![DirectML::default().build(), CPU::default().build()],
            "DirectML -> CPU".to_string(),
        ),
        "cpu" => (vec![CPU::default().build()], "CPU".to_string()),
        _ => {
            #[allow(unused_mut)]
            let mut list: Vec<ExecutionProviderDispatch> = Vec::new();
            #[allow(unused_mut)]
            let mut names: Vec<&'static str> = Vec::new();

            #[cfg(feature = "cuda")]
            {
                list.push(TensorRT::default().build());
                list.push(CUDA::default().build());
                names.push("TRT/CUDA");
            }
            #[cfg(target_os = "windows")]
            {
                list.push(DirectML::default().build());
                names.push("DirectML");
            }
            list.push(CPU::default().build());
            names.push("CPU");

            (list, format!("AUTO ({})", names.join("/")))
        }
    }
}

fn inspect_model_input(session: &Session, default_size: usize) -> (String, TensorLayout, usize) {
    if let Some(first_input) = session.inputs().first() {
        let name = first_input.name().to_string();
        if let ValueType::Tensor { shape, .. } = first_input.dtype() {
            let dims: &[i64] = shape.as_ref();
            if dims.len() == 4 {
                if dims[1] == 3 {
                    let sz = (dims[2].max(1) as usize).max(default_size.min(64));
                    return (
                        name,
                        TensorLayout::Nchw,
                        if dims[2] > 0 { dims[2] as usize } else { sz },
                    );
                } else if dims[3] == 3 {
                    let sz = if dims[1] > 0 {
                        dims[1] as usize
                    } else {
                        default_size
                    };
                    return (name, TensorLayout::Nhwc, sz);
                }
            }
        }
        return (name, TensorLayout::Nchw, default_size);
    }
    ("input".to_string(), TensorLayout::Nchw, default_size)
}

fn generate_palm_ssd_anchors(input_size: usize) -> Vec<SsdAnchor> {
    let layer_specs: [(usize, usize); 2] = [(8, 2), (16, 6)];
    let mut anchors = Vec::with_capacity(2016);

    for (stride, anchors_per_cell) in layer_specs {
        let grid_h = input_size / stride;
        let grid_w = input_size / stride;
        for y in 0..grid_h {
            let cy = (y as f32 + 0.5) / (grid_h as f32);
            for x in 0..grid_w {
                let cx = (x as f32 + 0.5) / (grid_w as f32);
                for _ in 0..anchors_per_cell {
                    anchors.push(SsdAnchor { x: cx, y: cy });
                }
            }
        }
    }
    anchors
}

fn non_maximum_suppression(mut boxes: Vec<PalmDetection>, iou_threshold: f32) -> Vec<PalmDetection> {
    boxes.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut selected: Vec<PalmDetection> = Vec::new();
    for cand in boxes {
        let overlaps = selected
            .iter()
            .any(|kept| compute_iou(kept, &cand) > iou_threshold);
        if !overlaps {
            selected.push(cand);
        }
    }
    selected
}

#[inline]
fn compute_iou(a: &PalmDetection, b: &PalmDetection) -> f32 {
    let inter_xmin = a.xmin.max(b.xmin);
    let inter_ymin = a.ymin.max(b.ymin);
    let inter_xmax = a.xmax.min(b.xmax);
    let inter_ymax = a.ymax.min(b.ymax);

    let inter_w = (inter_xmax - inter_xmin).max(0.0);
    let inter_h = (inter_ymax - inter_ymin).max(0.0);
    let inter_area = inter_w * inter_h;
    if inter_area <= 0.0 {
        return 0.0;
    }

    let area_a = ((a.xmax - a.xmin).max(0.0)) * ((a.ymax - a.ymin).max(0.0));
    let area_b = ((b.xmax - b.xmin).max(0.0)) * ((b.ymax - b.ymin).max(0.0));
    let union = area_a + area_b - inter_area;
    if union <= 1.0e-9 {
        0.0
    } else {
        inter_area / union
    }
}

fn palm_detection_to_oriented_roi(det: &PalmDetection, letterbox: &LetterboxMeta) -> OrientedRoi {
    let norm_cx = 0.5 * (det.xmin + det.xmax);
    let norm_cy = 0.5 * (det.ymin + det.ymax);
    let norm_w = (det.xmax - det.xmin).abs();
    let norm_h = (det.ymax - det.ymin).abs();

    let (palm_cx_px, palm_cy_px) = letterbox.unmap_to_pixels(norm_cx, norm_cy);
    let (palm_w_px, palm_h_px) = letterbox.unmap_size_to_pixels(norm_w, norm_h);
    let palm_size_px = palm_w_px.max(palm_h_px).max(24.0);

    let (kp0_x, kp0_y) = letterbox.unmap_to_pixels(det.keypoints[0].0, det.keypoints[0].1);
    let (kp2_x, kp2_y) = letterbox.unmap_to_pixels(det.keypoints[2].0, det.keypoints[2].1);

    let dx = kp2_x - kp0_x;
    let dy = kp2_y - kp0_y;
    let rotation_rad = normalize_radians(FRAC_PI_2 - (-dy).atan2(dx));

    let shift_px = PALM_SHIFT_ALONG_HAND * palm_size_px;
    let center_x = palm_cx_px + shift_px * rotation_rad.sin();
    let center_y = palm_cy_px - shift_px * rotation_rad.cos();
    let size_px = palm_size_px * PALM_ENLARGE_SCALE;

    OrientedRoi {
        center_x,
        center_y,
        size_px,
        rotation_rad,
    }
}

fn parse_landmark_scalars(scalars: &[(String, f32)]) -> (f32, f32) {
    if scalars.is_empty() {
        return (1.0, 0.5);
    }
    let mut presence = None;
    let mut handedness = None;

    for (name, val) in scalars {
        let prob = normalize_probability(*val);
        if name.contains("handflag")
            || name.contains("score")
            || name.contains("presence")
            || name.contains("conf")
        {
            presence = Some(prob);
        } else if name.contains("handed") || name.contains("lr") {
            handedness = Some(prob);
        }
    }

    let p = presence.unwrap_or_else(|| normalize_probability(scalars[0].1));
    let h = handedness.unwrap_or_else(|| {
        if scalars.len() >= 2 {
            normalize_probability(scalars[1].1)
        } else {
            0.5
        }
    });
    (p, h)
}

#[inline]
fn normalize_probability(v: f32) -> f32 {
    if (0.0..=1.0).contains(&v) {
        v
    } else {
        let clamped = v.clamp(-80.0, 80.0);
        1.0 / (1.0 + (-clamped).exp())
    }
}

#[inline]
fn max_abs(slice: &[f32]) -> f32 {
    slice.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_models_load_and_infer() {
        let palm = Path::new("assets/models/palm_detection_full.onnx");
        let hand = Path::new("assets/models/hand_landmark_full.onnx");
        if !palm.exists() || !hand.exists() {
            return;
        }

        let mut engine = OnnxVisionEngine::new(palm, hand, "auto").unwrap();
        let dummy_rgb = vec![128u8; 640 * 480 * 3];

        let res = engine.infer_hand(&dummy_rgb, 640, 480, true).unwrap();
        assert!(res.is_none());

        let test_roi = OrientedRoi {
            center_x: 320.0,
            center_y: 240.0,
            size_px: 224.0,
            rotation_rad: 0.0,
        };
        let _stage2 = engine
            .run_landmark_stage(&dummy_rgb, 640, 480, &test_roi, true, false)
            .unwrap();
    }
}
