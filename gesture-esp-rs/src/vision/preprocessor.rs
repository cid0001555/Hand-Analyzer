use crate::vision::skeleton::OrientedRoi;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    Nchw,
    Nhwc,
}

#[derive(Debug, Clone, Copy)]
pub struct LetterboxMeta {
    pub src_w: f32,
    pub src_h: f32,
    pub pad_x_norm: f32,
    pub pad_y_norm: f32,
    pub scale_x_norm: f32,
    pub scale_y_norm: f32,
}

impl LetterboxMeta {
    #[inline]
    pub fn unmap_to_pixels(&self, norm_x: f32, norm_y: f32) -> (f32, f32) {
        let unpadded_x = (norm_x - self.pad_x_norm) / self.scale_x_norm.max(1.0e-6);
        let unpadded_y = (norm_y - self.pad_y_norm) / self.scale_y_norm.max(1.0e-6);
        (unpadded_x * self.src_w, unpadded_y * self.src_h)
    }

    #[inline]
    pub fn unmap_size_to_pixels(&self, norm_w: f32, norm_h: f32) -> (f32, f32) {
        let pw = (norm_w / self.scale_x_norm.max(1.0e-6)) * self.src_w;
        let ph = (norm_h / self.scale_y_norm.max(1.0e-6)) * self.src_h;
        (pw, ph)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AffineCropInverse {
    pub center_x_px: f32,
    pub center_y_px: f32,
    pub size_px: f32,
    pub cos_rot: f32,
    pub sin_rot: f32,
    pub inv_src_w: f32,
    pub inv_src_h: f32,
}

impl AffineCropInverse {
    #[inline]
    pub fn map_crop_norm_to_frame_norm(&self, crop_u: f32, crop_v: f32) -> (f32, f32) {
        let local_x = (crop_u - 0.5) * self.size_px;
        let local_y = (crop_v - 0.5) * self.size_px;

        let src_px = self.center_x_px + (local_x * self.cos_rot - local_y * self.sin_rot);
        let src_py = self.center_y_px + (local_x * self.sin_rot + local_y * self.cos_rot);

        (src_px * self.inv_src_w, src_py * self.inv_src_h)
    }
}

pub fn letterbox_rgb_to_tensor(
    rgb: &[u8],
    src_w: u32,
    src_h: u32,
    dst_size: usize,
    layout: TensorLayout,
) -> (Vec<f32>, LetterboxMeta) {
    let sw = src_w as f32;
    let sh = src_h as f32;
    let dst_f = dst_size as f32;

    let scale = (dst_f / sw).min(dst_f / sh);
    let resized_w = (sw * scale).round() as usize;
    let resized_h = (sh * scale).round() as usize;

    let pad_left = (dst_size.saturating_sub(resized_w)) / 2;
    let pad_top = (dst_size.saturating_sub(resized_h)) / 2;

    let plane = dst_size * dst_size;
    let mut tensor = vec![0.0f32; plane * 3];

    let inv_scale = 1.0 / scale.max(1.0e-6);
    let inv_255 = 1.0 / 255.0;

    for dy in 0..resized_h {
        let sy = ((dy as f32 + 0.5) * inv_scale - 0.5).clamp(0.0, sh - 1.0);
        let out_y = dy + pad_top;
        if out_y >= dst_size {
            continue;
        }
        for dx in 0..resized_w {
            let sx = ((dx as f32 + 0.5) * inv_scale - 0.5).clamp(0.0, sw - 1.0);
            let out_x = dx + pad_left;
            if out_x >= dst_size {
                continue;
            }
            let (r, g, b) = sample_rgb_bilinear(rgb, src_w as usize, src_h as usize, sx, sy);
            write_tensor_pixel(
                &mut tensor,
                dst_size,
                out_x,
                out_y,
                r * inv_255,
                g * inv_255,
                b * inv_255,
                layout,
            );
        }
    }

    let meta = LetterboxMeta {
        src_w: sw,
        src_h: sh,
        pad_x_norm: (pad_left as f32) / dst_f,
        pad_y_norm: (pad_top as f32) / dst_f,
        scale_x_norm: (resized_w as f32) / dst_f,
        scale_y_norm: (resized_h as f32) / dst_f,
    };

    (tensor, meta)
}

pub fn extract_oriented_roi_tensor(
    rgb: &[u8],
    src_w: u32,
    src_h: u32,
    roi: &OrientedRoi,
    dst_size: usize,
    layout: TensorLayout,
) -> (Vec<f32>, AffineCropInverse) {
    let plane = dst_size * dst_size;
    let mut tensor = vec![0.0f32; plane * 3];

    let cos_rot = roi.rotation_rad.cos();
    let sin_rot = roi.rotation_rad.sin();
    let dst_f = dst_size as f32;
    let inv_dst = 1.0 / dst_f;
    let inv_255 = 1.0 / 255.0;

    let w_usize = src_w as usize;
    let h_usize = src_h as usize;

    for dy in 0..dst_size {
        let v = (dy as f32 + 0.5) * inv_dst - 0.5;
        let local_y = v * roi.size_px;

        for dx in 0..dst_size {
            let u = (dx as f32 + 0.5) * inv_dst - 0.5;
            let local_x = u * roi.size_px;

            let sx = roi.center_x + (local_x * cos_rot - local_y * sin_rot);
            let sy = roi.center_y + (local_x * sin_rot + local_y * cos_rot);

            if sx >= 0.0 && sx < (src_w as f32) && sy >= 0.0 && sy < (src_h as f32) {
                let (r, g, b) = sample_rgb_bilinear(rgb, w_usize, h_usize, sx, sy);
                write_tensor_pixel(
                    &mut tensor,
                    dst_size,
                    dx,
                    dy,
                    r * inv_255,
                    g * inv_255,
                    b * inv_255,
                    layout,
                );
            }
        }
    }

    let inverse = AffineCropInverse {
        center_x_px: roi.center_x,
        center_y_px: roi.center_y,
        size_px: roi.size_px,
        cos_rot,
        sin_rot,
        inv_src_w: 1.0 / (src_w.max(1) as f32),
        inv_src_h: 1.0 / (src_h.max(1) as f32),
    };

    (tensor, inverse)
}

#[inline]
fn write_tensor_pixel(
    tensor: &mut [f32],
    size: usize,
    x: usize,
    y: usize,
    r: f32,
    g: f32,
    b: f32,
    layout: TensorLayout,
) {
    match layout {
        TensorLayout::Nchw => {
            let plane = size * size;
            let idx = y * size + x;
            tensor[idx] = r;
            tensor[plane + idx] = g;
            tensor[2 * plane + idx] = b;
        }
        TensorLayout::Nhwc => {
            let base = (y * size + x) * 3;
            tensor[base] = r;
            tensor[base + 1] = g;
            tensor[base + 2] = b;
        }
    }
}

#[inline]
fn sample_rgb_bilinear(rgb: &[u8], width: usize, height: usize, x: f32, y: f32) -> (f32, f32, f32) {
    let x_clamped = x.clamp(0.0, (width.saturating_sub(1)) as f32);
    let y_clamped = y.clamp(0.0, (height.saturating_sub(1)) as f32);

    let x0 = x_clamped.floor() as usize;
    let y0 = y_clamped.floor() as usize;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));

    let fx = x_clamped - (x0 as f32);
    let fy = y_clamped - (y0 as f32);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let i00 = (y0 * width + x0) * 3;
    let i10 = (y0 * width + x1) * 3;
    let i01 = (y1 * width + x0) * 3;
    let i11 = (y1 * width + x1) * 3;

    let r = (rgb[i00] as f32) * w00
        + (rgb[i10] as f32) * w10
        + (rgb[i01] as f32) * w01
        + (rgb[i11] as f32) * w11;
    let g = (rgb[i00 + 1] as f32) * w00
        + (rgb[i10 + 1] as f32) * w10
        + (rgb[i01 + 1] as f32) * w01
        + (rgb[i11 + 1] as f32) * w11;
    let b = (rgb[i00 + 2] as f32) * w00
        + (rgb[i10 + 2] as f32) * w10
        + (rgb[i01 + 2] as f32) * w01
        + (rgb[i11 + 2] as f32) * w11;

    (r, g, b)
}
