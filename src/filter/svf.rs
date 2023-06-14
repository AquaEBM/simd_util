use super::{*, simd_util::map, smoothing::*};

use std::f32::consts::PI;

#[derive(Default)]
/// Digital implementation of the analogue SVF Filter, with built-in
/// parameter smoothing. Based on the one in the book The Art of VA
/// Filter Design by Vadim Zavalinish
/// 
/// Capable of outputing many different filter types,
/// (highpass, lowpass, bandpass, notch, shelving....)
pub struct SVF<const N: usize>
where
    LaneCount<N>: SupportedLaneCount
{
    g: LogSmoother<N>,
    r: LogSmoother<N>,
    k: LogSmoother<N>,
    s: [Integrator<N> ; 2],
    pi_tick: Simd<f32, N>,
    x: Simd<f32, N>,
    hp: Simd<f32, N>,
    bp: Simd<f32, N>,
    lp: Simd<f32, N>,
}

impl<const N: usize> SVF<N>
where
    LaneCount<N>: SupportedLaneCount
{
    pub fn set_sample_rate(&mut self, sr: f32) {
        self.pi_tick = Simd::splat(PI / sr);
    }

    pub fn reset(&mut self) {
        self.s[0].reset();
        self.s[1].reset();
    }

    /// _Immediately_ sets the filters cutoff frequency to `cutoff`
    /// 
    /// For a version that smooths it towards the given value, see `Self::set_cutoff_smoothed`
    pub fn set_cutoff(&mut self, cutoff: Simd<f32, N>) {
        *self.g = map(cutoff * self.pi_tick, f32::tan);
    }

    /// Smoothes the filter's cutoff frequency towards
    /// `cutoff`, effectively reaching it after `block_len` samples
    pub fn set_cutoff_smoothed(&mut self, cutoff: Simd<f32, N>, block_len: usize) {

        self.g.set_target(map(cutoff * self.pi_tick, f32::tan), block_len)
    }

    /// _Immediately_ sets the filters resonance amount to `res`
    /// 
    /// For a version that smooths it towards the given value, see `Self::set_resonance_smoothed`
    pub fn set_resonance(&mut self, res: Simd<f32, N>) {
        *self.r = res;
    }

    /// Smoothes the filter's resonance amount towards
    /// `res`, effectively reaching it after `block_len` samples
    pub fn set_resonance_smoothed(&mut self, res: Simd<f32, N>, block_len: usize) {
        self.r.set_target(res, block_len)
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
    pub fn set_params(&mut self, cutoff: Simd<f32, N>, res: Simd<f32, N>, gain: Simd<f32, N>) {
        self.set_cutoff(cutoff);
        self.set_gain(gain);
        self.set_resonance(res);
    }

    /// Convenience method to smooth all parameters toward the given values
    /// effectively reaching them after `block_len` samples
    pub fn set_params_smoothed(
        &mut self,
        cutoff: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>,
        block_len: usize,
    ) {
        self.set_cutoff_smoothed(cutoff, block_len);
        self.set_resonance_smoothed(res, block_len);
        self.set_gain_smoothed(gain, block_len);
    }

    /// Updates the cutoff parameter smoother.
    /// 
    /// After calling `Self::set_cutoff_smoothed(value, num_samples)` this should
    /// be called only once per sample, _up to_ `num_samples` times, until
    /// `Self::set_cutoff_smoothed` is to be called again
    pub fn update_cutoff_smoother(&mut self) {
        self.g.tick()
    }

    /// Updates the resonance parameter smoother.
    /// 
    /// After calling `Self::set_resonance_smoothed(value, num_samples)` this should
    /// be called only _once_ per sample, _up to_ `num_samples` times, until
    /// `Self::set_resonance_smoothed` is to be called again
    pub fn update_resonance_smoother(&mut self) {
        self.r.tick()
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
    pub fn update_all_smoothers(&mut self) {
        self.update_cutoff_smoother();
        self.update_gain_smoother();
        self.update_resonance_smoother();
    }

    /// Update the filter's internal state, given the provided input sample.
    /// 
    /// This should be called _only once_ per sample, _every sample_
    /// 
    /// After calling this, you can get different filter outputs
    /// using `Self::get_{highpass, bandpass, notch, ...}`
    pub fn process(&mut self, sample: Simd<f32, N>) {

        let g = *self.g;

        let g1 = *self.r + g;

        self.x = sample;
        self.hp = (sample - *self.s[1] - *self.s[0] * g1) / (Simd::splat(1.) + g * g1);
        self.bp = self.s[0].process(self.hp, g);
        self.lp = self.s[1].process(self.bp, g);
    }

    /// Get the current (smoothed?) value of the internal gain parameter
    pub fn get_gain(&self) -> Simd<f32, N> {
        *self.k
    }

    pub fn get_highpass(&self) -> Simd<f32, N> {
        *self.k * self.hp
    }

    pub fn get_bandpass(&self) -> Simd<f32, N> {
        *self.k * self.bp
    }

    pub fn get_lowpass(&self) -> Simd<f32, N> {
        *self.k * self.lp
    }

    pub fn get_bandpass1(&self) -> Simd<f32, N> {
        *self.r * self.bp * *self.k
    }

    pub fn get_allpass(&self) -> Simd<f32, N> {
        let bp1 = *self.r * self.bp;
        *self.k * Simd::splat(2.).mul_add(bp1, -self.x)
    }

    pub fn get_notch(&self) -> Simd<f32, N> {
        *self.k * self.r.mul_add(-self.bp, self.x)
    }

    pub fn get_peaking(&self) -> Simd<f32, N> {
       *self.k * (self.lp - self.hp)
    }

    pub fn get_high_shelf(&self) -> Simd<f32, N> {

        let hp = self.hp;
        self.k.mul_add(hp, self.x - hp)
    }

    pub fn get_band_shelf(&self) -> Simd<f32, N> {

        let bp1 = *self.r * self.bp;
        self.k.mul_add(bp1, self.x - bp1)
    }

    pub fn get_low_shelf(&self) -> Simd<f32, N> {
        let lp = self.lp;
        self.k.mul_add(lp, self.x - lp)
    }
}