use super::{*, simd_util::map, smoothing::*};

use std::f32::consts::PI;

#[derive(Default)]
pub struct OnePole<const N: usize>
where
    LaneCount<N>: SupportedLaneCount
{
    g: LogSmoother<N>,
    k: LogSmoother<N>,
    s: Integrator<N>,
    lp: Simd<f32, N>,
    hp: Simd<f32, N>,
    pi_tick: Simd<f32, N>,
}

impl<const N: usize> OnePole<N>
where
    LaneCount<N>: SupportedLaneCount
{
    pub fn reset(&mut self) {
        *self = Self {
            pi_tick: self.pi_tick,
            ..Default::default()
        }
    }

    pub fn set_sample_rate(&mut self, sr: f32) {
        self.pi_tick = Simd::splat(PI / sr);
    }

    /// _Immediately_ sets the filters cutoff frequency to `cutoff`
    /// 
    /// For a version that smooths it towards the given value, see `Self::set_cutoff_smoothed`
    pub fn set_cutoff(&mut self, cutoff: Simd<f32, N>) {
        let g = map(cutoff * self.pi_tick, f32::tan);

        *self.g = g / (Simd::splat(1.) + g);
    }

    /// Smoothes the filter's cutoff frequency towards
    /// `cutoff`, effectively reaching it after `block_len` samples
    pub fn set_cutoff_smoothed(&mut self, cutoff: Simd<f32, N>, block_len: usize) {
        let g = map(cutoff * self.pi_tick, f32::tan);

        self.g.set_target(
            g / (Simd::splat(1.) + g),
            block_len
        );
    }

    /// _Immediately_ sets the filters gain to `k`
    /// 
    /// For a version that smooths it towards the given value, see `Self::set_gain_smoothed`
    pub fn set_gain(&mut self, k: Simd<f32, N>) {
        *self.k = k;
    }

    /// Smoothes the filter's gain towards
    /// `k`, effectively reaching it after `block_len` samples
    pub fn set_gain_smoothed(&mut self, k: Simd<f32, N>, block_len: usize) {
        self.k.set_target(k, block_len)
    }

    /// Convenience method to _immediately_ set all parameters
    /// 
    /// For a smoothed version of this, see 'Self::set_params_smoothed`
    pub fn set_params(&mut self, cutoff: Simd<f32, N>, k: Simd<f32, N>) {
        self.set_cutoff(cutoff);
        self.set_gain(k);
    }

    /// Convenience method to smooth all parameters toward the given values
    /// effectively reaching them after `block_len` samples
    pub fn set_params_smoothed(
        &mut self,
        cutoff: Simd<f32, N>,
        k: Simd<f32, N>,
        block_len: usize
    ) {
        self.set_cutoff_smoothed(cutoff, block_len);
        self.set_gain_smoothed(k, block_len);
    }

    /// Updates the cutoff parameter smoother.
    /// 
    /// After calling `Self::set_cutoff_smoothed(value, num_samples)` this should
    /// be called only once per sample, _up to_ `num_samples` times, until
    /// `Self::set_cutoff_smoothed` is to be called again
    pub fn update_cutoff_smoother(&mut self) {
        self.g.tick()
    }

    /// Updates the gain parameter smoother.
    /// 
    /// After calling `Self::set_gain_smoothed(value, num_samples)` this should
    /// be called only _once_ per sample, _up to_ `num_samples` times, until
    /// `Self::set_gain_smoothed` is to be called again
    pub fn update_gain_smoother(&mut self) {
        self.k.tick()
    }

    /// convenience method to update all the filter's internal parameter smoothers at once.
    /// 
    /// After calling `Self::set_params_smoothed(values, ..., num_samples)` this should
    /// be called only _once_ per sample, _up to_ `num_samples` times, until
    /// `Self::set_params_smoothed` is to be called again
    pub fn update_smoothers(&mut self) {
        self.update_cutoff_smoother();
        self.update_gain_smoother();
    }

    /// Update the filter's internal state, given the provided input sample.
    /// 
    /// This should be called _only once_ per sample, _every sample_
    /// 
    /// After calling this, you can get different filter outputs
    /// using `Self::get_{highpass, lowpass, allpass, ...}`
    pub fn process(&mut self, sample: Simd<f32, N>) {

        let lp = self.s.process(sample - *self.s, *self.g);

        self.hp = sample - self.lp;
        self.lp = lp;
    }

    /// Get the current (smoothed?) value of the internal gain parameter
    pub fn get_gain(&mut self) -> Simd<f32, N> {
        *self.k
    }

    pub fn get_highpass(&self) -> Simd<f32, N> {
        *self.k * self.hp
    }

    pub fn get_lowpass(&self) -> Simd<f32, N> {
        *self.k * self.lp
    }

    pub fn get_allpass(&self) -> Simd<f32, N> {
        *self.k * (self.lp - self.hp)
    }

    pub fn get_lowshelf(&self) -> Simd<f32, N> {
        self.k.mul_add(self.lp, self.hp)
    }

    pub fn get_highshelf(&self) -> Simd<f32, N> {
        self.k.mul_add(self.hp, self.lp)
    }
}
