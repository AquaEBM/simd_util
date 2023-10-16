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
    max_cutoff: Simd<f32, N>,
    min_tick: Simd<f32, N>,
    pi_tick: Simd<f32, N>,
}

impl<const N: usize> OnePole<N>
where
    LaneCount<N>: SupportedLaneCount
{
    pub fn reset(&mut self) {

        *self = Self {
            max_cutoff: self.max_cutoff,
            min_tick: self.min_tick,
            pi_tick: self.pi_tick,
            ..Default::default()
        }
    }

    pub fn set_sample_rate(&mut self, sr: f32) {

        self.max_cutoff = Simd::splat(sr * MAX_CUTOFF_RATIO);
        self.min_tick = Simd::splat(1. / sr * MAX_CUTOFF_RATIO);
        self.pi_tick = Simd::splat(PI / sr);
    }

    fn pre_gain_from_cutoff(&self, cutoff: Simd<f32, N>) -> Simd<f32, N> {

        let g = map(cutoff * self.pi_tick, f32::tan);

        g / (Simd::splat(1.) + g)
    }

    /// this pretty much only sets the filter's cutoff to the
    /// given value and the gain to `1.0`, call this _only_ if you intend to
    /// output non-shelving filter shapes.
    pub fn set_cutoff(&mut self, cutoff: Simd<f32, N>) {
        self.g.set_instantly(self.pre_gain_from_cutoff(cutoff));
        self.k.set_instantly(Simd::splat(1.));
    }

    /// call this _only_ if you intend to output low-shelving filter shapes.
    pub fn set_params_low_shelving(&mut self, cutoff: Simd<f32, N>, gain: Simd<f32, N>) {

        let ratio = cutoff * self.min_tick;

        let k = gain.simd_max(ratio * ratio);

        self.k.set_instantly(k);
        self.g.set_instantly(self.pre_gain_from_cutoff(cutoff / k.sqrt()));
    }

    /// call this _only_ if you intend to output high-shelving filter shapes.
    pub fn set_params_high_shelving(&mut self, cutoff: Simd<f32, N>, gain: Simd<f32, N>) {

        let ratio = self.max_cutoff / cutoff;

        let k = gain.simd_min(ratio * ratio);

        self.k.set_instantly(k);
        self.g.set_instantly(self.pre_gain_from_cutoff(cutoff * k.sqrt()));
    }

    /// like `Self::set_cutoff` but smoothed
    pub fn set_cutoff_smoothed(
        &mut self,
        cutoff: Simd<f32, N>,
        block_len: usize
    ) {
        self.k.set_target(Simd::splat(1.), block_len);

        self.g.set_target(
            self.pre_gain_from_cutoff(cutoff),
            block_len
        );
    }

    /// like `Self::set_params_low_shelving` but smoothed
    pub fn set_params_low_shelving_smoothed(
        &mut self,
        cutoff: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize
    ) {

        let ratio = cutoff * self.min_tick;

        let k = gain.simd_max(ratio * ratio);

        self.k.set_target(k, num_samples);

        self.g.set_target(
            self.pre_gain_from_cutoff(cutoff / k.sqrt()),
            num_samples
        );
    }

    /// like `Self::set_params_high_shelving` but smoothed.
    pub fn set_params_high_shelving_smoothed(
        &mut self,
        cutoff: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize
    ) {

        let ratio = self.max_cutoff / cutoff;

        let k = gain.simd_min(ratio * ratio);

        self.k.set_target(k, num_samples);

        self.g.set_target(
            self.pre_gain_from_cutoff(cutoff * k.sqrt()),
            num_samples
        );
    }

    ///update t.set_instantly(filter's internal parameter smoothers.
    /// 
    /// After calling `Self::set_params_smoothed(values, ..., num_samples)` this should
    /// be called only _once_ per sample, _up to_ `num_samples` times, until
    /// `Self::set_params_smoothed` is to be called again
    pub fn update_smoothers(&mut self) {
        self.g.tick();
        self.k.tick();
    }

    /// Update the filter's internal state, given the provided input sample.
    /// 
    /// This should be called _only once_ per sample, _every sample_
    /// 
    /// After calling this, you can get different filter outputs
    /// using `Self::get_{highpass, lowpass, allpass, ...}`
    pub fn process(&mut self, sample: Simd<f32, N>) {

        let lp = self.s.process(sample - *self.s, self.g.get_current());

        self.hp = sample - self.lp;
        self.lp = lp;
    }

    pub fn get_highpass(&self) -> Simd<f32, N> {
        self.hp
    }

    pub fn get_lowpass(&self) -> Simd<f32, N> {
        self.lp
    }

    pub fn get_allpass(&self) -> Simd<f32, N> {
        self.lp - self.hp
    }

    pub fn get_lowshelf(&self) -> Simd<f32, N> {
        self.k.get_current().mul_add(self.lp, self.hp)
    }

    pub fn get_highshelf(&self) -> Simd<f32, N> {
        
        self.k.get_current().mul_add(self.hp, self.lp)
    }
}
