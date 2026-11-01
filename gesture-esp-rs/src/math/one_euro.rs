use std::f32::consts::TAU;
use glam::Vec3A;

use crate::config::FilterConfig;

const MIN_DT_SEC: f32 = 1.0e-5;
const MAX_DT_SEC: f32 = 0.25;

#[inline]
pub fn smoothing_factor(dt_sec: f32, cutoff_hz: f32) -> f32 {
    let r = TAU * dt_sec * cutoff_hz.max(1.0e-4);
    r / (1.0 + r)
}

#[derive(Debug, Clone, Copy)]
pub struct OneEuroFilterScalar {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x_prev: f32,
    dx_prev: f32,
    initialized: bool,
}

impl OneEuroFilterScalar {
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            min_cutoff: min_cutoff.max(1.0e-4),
            beta: beta.max(0.0),
            d_cutoff: d_cutoff.max(1.0e-4),
            x_prev: 0.0,
            dx_prev: 0.0,
            initialized: false,
        }
    }

    pub fn reset(&mut self) {
        self.x_prev = 0.0;
        self.dx_prev = 0.0;
        self.initialized = false;
    }

    pub fn update_params(&mut self, min_cutoff: f32, beta: f32, d_cutoff: f32) {
        self.min_cutoff = min_cutoff.max(1.0e-4);
        self.beta = beta.max(0.0);
        self.d_cutoff = d_cutoff.max(1.0e-4);
    }

    pub fn filter_with_velocity(&mut self, x: f32, dt_sec: f32) -> (f32, f32) {
        if !x.is_finite() {
            return (self.x_prev, self.dx_prev);
        }
        if !self.initialized {
            self.x_prev = x;
            self.dx_prev = 0.0;
            self.initialized = true;
            return (x, 0.0);
        }

        let te = dt_sec.clamp(MIN_DT_SEC, MAX_DT_SEC);
        let dx_raw = (x - self.x_prev) / te;
        let alpha_d = smoothing_factor(te, self.d_cutoff);
        let dx_hat = alpha_d * dx_raw + (1.0 - alpha_d) * self.dx_prev;

        let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
        let alpha = smoothing_factor(te, cutoff);
        let x_hat = alpha * x + (1.0 - alpha) * self.x_prev;

        self.x_prev = x_hat;
        self.dx_prev = dx_hat;
        (x_hat, dx_hat)
    }

    #[inline]
    pub fn filter(&mut self, x: f32, dt_sec: f32) -> f32 {
        self.filter_with_velocity(x, dt_sec).0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OneEuroFilterVec3A {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x_prev: Vec3A,
    dx_prev: Vec3A,
    initialized: bool,
}

impl OneEuroFilterVec3A {
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            min_cutoff: min_cutoff.max(1.0e-4),
            beta: beta.max(0.0),
            d_cutoff: d_cutoff.max(1.0e-4),
            x_prev: Vec3A::ZERO,
            dx_prev: Vec3A::ZERO,
            initialized: false,
        }
    }

    pub fn from_config(cfg: FilterConfig) -> Self {
        Self::new(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
    }

    pub fn reset(&mut self) {
        self.x_prev = Vec3A::ZERO;
        self.dx_prev = Vec3A::ZERO;
        self.initialized = false;
    }

    pub fn update_params(&mut self, min_cutoff: f32, beta: f32, d_cutoff: f32) {
        self.min_cutoff = min_cutoff.max(1.0e-4);
        self.beta = beta.max(0.0);
        self.d_cutoff = d_cutoff.max(1.0e-4);
    }

    pub fn filter(&mut self, x_t: Vec3A, dt_sec: f32) -> Vec3A {
        if !x_t.is_finite() {
            return self.x_prev;
        }

        if !self.initialized {
            self.x_prev = x_t;
            self.dx_prev = Vec3A::ZERO;
            self.initialized = true;
            return x_t;
        }

        let te = dt_sec.clamp(MIN_DT_SEC, MAX_DT_SEC);

        let dx_raw = (x_t - self.x_prev) / te;
        let alpha_d = smoothing_factor(te, self.d_cutoff);
        let dx_hat = alpha_d * dx_raw + (1.0 - alpha_d) * self.dx_prev;

        let speed = dx_hat.length();
        let fc = self.min_cutoff + self.beta * speed;

        let alpha = smoothing_factor(te, fc);
        let x_hat = alpha * x_t + (1.0 - alpha) * self.x_prev;

        self.x_prev = x_hat;
        self.dx_prev = dx_hat;
        x_hat
    }

    #[inline]
    pub fn velocity(&self) -> Vec3A {
        self.dx_prev
    }
}

#[derive(Debug, Clone)]
pub struct SkeletonFilterBank {
    pub filters: [OneEuroFilterVec3A; 21],
}

impl SkeletonFilterBank {
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            filters: std::array::from_fn(|_| OneEuroFilterVec3A::new(min_cutoff, beta, d_cutoff)),
        }
    }

    pub fn from_config(cfg: FilterConfig) -> Self {
        Self::new(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)
    }

    pub fn update_params(&mut self, cfg: FilterConfig) {
        for f in &mut self.filters {
            f.update_params(cfg.min_cutoff, cfg.beta, cfg.d_cutoff);
        }
    }

    pub fn reset(&mut self) {
        for f in &mut self.filters {
            f.reset();
        }
    }

    pub fn filter_skeleton(&mut self, raw_joints: &[Vec3A; 21], dt_sec: f32) -> [Vec3A; 21] {
        let mut out = [Vec3A::ZERO; 21];
        for i in 0..21 {
            out[i] = self.filters[i].filter(raw_joints[i], dt_sec);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_euro_filter() {
        let mut filter = OneEuroFilterVec3A::new(0.8, 0.5, 1.0);
        let dt = 1.0 / 60.0;

        let base = Vec3A::new(0.5, 0.5, 0.0);
        let mut last = filter.filter(base, dt);
        for i in 0..30 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let noisy = base + Vec3A::splat(0.005 * sign);
            last = filter.filter(noisy, dt);
        }
        let jitter_residual = (last - base).length();
        assert!(jitter_residual < 0.004);

        let mut static_lpf = OneEuroFilterVec3A::new(0.8, 0.0, 1.0);
        let _ = static_lpf.filter(base, dt);

        let target = Vec3A::new(0.9, 0.9, 0.0);
        let mut last_static = base;
        for _ in 0..10 {
            last = filter.filter(target, dt);
            last_static = static_lpf.filter(target, dt);
        }
        let adaptive_err = (last - target).length();
        let static_err = (last_static - target).length();
        assert!(adaptive_err < 0.025 && adaptive_err < 0.25 * static_err);
    }
}
