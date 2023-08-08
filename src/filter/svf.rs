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

    fn pre_gain_from_cutoff(&self, cutoff: Simd<f32, N>) -> Simd<f32, N> {
        map(cutoff * self.pi_tick, f32::tan)
    }

    fn set_values(&mut self, cutoff: Simd<f32, N>, res: Simd<f32, N>, gain: Simd<f32, N>) {
        *self.g = self.pre_gain_from_cutoff(cutoff);
        *self.k = gain;
        *self.r = res;
    }

    fn set_values_smoothed(
        &mut self,
        cutoff: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize
    ) {
        self.g.set_target(self.pre_gain_from_cutoff(cutoff), num_samples);
        self.r.set_target(res, num_samples);
        self.k.set_target(gain, num_samples);
    }

    /// smooth parameters towards the given values, effectively
    /// reaching them after `num_samples` samples, call this if you
    /// intend to later output _only_ low-shelving filter shapes
    /// 
    /// For a smoothed version of this, see `Self::set_params_low_shelving_smoothed`
    pub fn set_params_low_shelving_smoothed(
        &mut self,
        cutoff: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize
    ) {
        let k = gain.sqrt();
        let m = k.sqrt();
        self.set_values_smoothed(cutoff * m, res, Simd::splat(1.) / k, num_samples);
    }

    /// smooth parameters towards the given values, effectively
    /// reaching them after `num_samples` samples, call this if you
    /// intend to later output _only_ band-shelving filter shapes
    /// 
    /// For a smoothed version of this, see `Self::set_params_band_shelving_smoothed`
    pub fn set_params_band_shelving_smoothed(
        &mut self,
        cutoff: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize
    ) {
        self.set_values_smoothed(cutoff, res * gain.recip().sqrt(), gain, num_samples);
    }

    /// smooth parameters towards the given values, effectively
    /// reaching them after `num_samples` samples, call this if you
    /// intend to later output _only_ high-shelving filter shapes
    /// 
    /// For a smoothed version of this, see `Self::set_params_high_or_band_shelving_smoothed`
    pub fn set_params_high_shelving_smoothed(
        &mut self,
        cutoff: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize
    ) {
        let k = gain.sqrt();
        let m = k.sqrt();
        self.set_values_smoothed(cutoff, res * m, k, num_samples);
    }

    /// smooth parameters towards the given values, effectively
    /// reaching them after `num_samples` samples, call this if you
    /// intend to later output non-shelving filter shapes
    /// 
    /// For a smoothed version of this, see `Self::set_params_non_shelving_smoothed`
    pub fn set_params_non_shelving_smoothed(
        &mut self, cutoff: Simd<f32, N>,
        res: Simd<f32, N>,
        num_samples: usize
    ) {
        self.g.set_target(self.pre_gain_from_cutoff(cutoff), num_samples);
        self.r.set_target(res, num_samples);
    }

    /// _immediately_ set parameters, call this if you
    /// intend to later output _only_ low-shelving filter shapes
    /// 
    /// For a smoothed version of this, see `Self::set_params_low_shelving_smoothed`
    pub fn set_params_low_shelving(
        &mut self,
        cutoff: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>
    ) {
        let k = gain.recip().sqrt();
        let m = k.sqrt();
        self.set_values(cutoff * m, res, k);
    }

    /// _immediately_ set parameters, call this if you
    /// intend to later output _only_ band-shelving filter shapes
    /// 
    /// For a smoothed version of this, see `Self::set_params_band_shelving_smoothed`
    pub fn set_params_band_shelving(
        &mut self,
        cutoff: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>
    ) {
        self.set_values(cutoff, res * gain.recip().sqrt(), gain);
    }

    /// _immediately_ set parameters, call this if you
    /// intend to later output _only_ high-shelving filter shapes
    /// 
    /// For a smoothed version of this, see `Self::set_params_high_or_band_shelving_smoothed`
    pub fn set_params_high_shelving(
        &mut self,
        cutoff: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>
    ) {
        let k = gain.sqrt();
        let m = k.sqrt();
        self.set_values(cutoff * m, res, k);
    }

    /// _immediately_ set parameters, call this if you
    /// intend to later output non-shelving filter shapes
    /// 
    /// For a smoothed version of this, see `Self::set_params_non_shelving_smoothed`
    pub fn set_params_non_shelving(&mut self, cutoff: Simd<f32, N>, res: Simd<f32, N>) {
        *self.g = self.pre_gain_from_cutoff(cutoff);
        *self.r = res;
    }

    /// convenience method to update all the filter's internal parameter smoothers at once.
    /// 
    /// After calling `Self::set_params_smoothed(values, ..., num_samples)` this should
    /// be called only _once_ per sample, _up to_ `num_samples` times, until
    /// `Self::set_params_smoothed` is to be called again
    pub fn update_all_smoothers(&mut self) {
        self.k.tick();
        self.r.tick();
        self.g.tick();
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
        self.hp = self.s[0].mul_add(-g1, sample - *self.s[1]) / g1.mul_add(g, Simd::splat(1.));
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
        self.bp
    }

    pub fn get_lowpass(&self) -> Simd<f32, N> {
        self.lp
    }

    pub fn get_bandpass1(&self) -> Simd<f32, N> {
        *self.r * self.bp
    }

    pub fn get_allpass(&self) -> Simd<f32, N> {
        Simd::splat(2.).mul_add(self.get_bandpass1(), -self.x)
    }

    pub fn get_notch(&self) -> Simd<f32, N> {
        self.r.mul_add(-self.bp, self.x)
    }

    pub fn get_high_shelf(&self) -> Simd<f32, N> {

        let m = *self.k;
        let bp1 = self.get_bandpass1();
        m.mul_add(m.mul_add(self.hp, bp1), self.lp)
    }

    pub fn get_band_shelf(&self) -> Simd<f32, N> {

        let bp1 = self.get_bandpass1();
        self.k.mul_add(bp1, self.x - bp1)
    }

    pub fn get_low_shelf(&self) -> Simd<f32, N> {
        let m = self.k.recip();
        let bp1 = self.get_bandpass1();
        m.mul_add(m.mul_add(self.lp, bp1), self.hp)
    }
}