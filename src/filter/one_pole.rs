use super::{simd_util::map, smoothing::*, *};

#[derive(Default)]
pub struct OnePole<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    g: LogSmoother<N>,
    k: LogSmoother<N>,
    s: Integrator<N>,
    lp: Simd<f32, N>,
    hp: Simd<f32, N>,
}

impl<const N: usize> OnePole<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    fn pre_gain_from_cutoff(w_c: Simd<f32, N>) -> Simd<f32, N> {
        let g = map(w_c * Simd::splat(0.5), f32::tan);

        g / (Simd::splat(1.) + g)
    }

    fn set_values(&mut self, g: Simd<f32, N>, k: Simd<f32, N>) {
        self.g.set_instantly(g);
        self.k.set_instantly(k);
    }

    /// this pretty much only sets the filter's cutoff to the
    /// given value and the gain to `1.0`, call this _only_ if you intend to
    /// output non-shelving filter shapes.
    pub fn set_params(&mut self, w_c: Simd<f32, N>, gain: Simd<f32, N>) {
        self.set_values(Self::pre_gain_from_cutoff(w_c), gain)
    }

    /// call this _only_ if you intend to output low-shelving filter shapes.
    pub fn set_params_low_shelving(&mut self, w_c: Simd<f32, N>, gain: Simd<f32, N>) {
        let g = Self::pre_gain_from_cutoff(w_c);

        self.k.set_instantly(gain);
        self.g.set_instantly(g / gain.sqrt());
    }

    /// call this _only_ if you intend to output high-shelving filter shapes.
    pub fn set_params_high_shelving(&mut self, w_c: Simd<f32, N>, gain: Simd<f32, N>) {
        let g = Self::pre_gain_from_cutoff(w_c);

        self.k.set_instantly(gain);
        self.g.set_instantly(g * gain.sqrt());
    }

    /// like `Self::set_params` but smoothed
    pub fn set_params_smoothed(&mut self, w_c: Simd<f32, N>, gain: Simd<f32, N>, block_len: usize) {
        self.k.set_target(gain, block_len);

        self.g
            .set_target(Self::pre_gain_from_cutoff(w_c), block_len);
    }

    /// like `Self::set_params_low_shelving` but smoothed
    pub fn set_params_low_shelving_smoothed(
        &mut self,
        w_c: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize,
    ) {
        let g = Self::pre_gain_from_cutoff(w_c);

        self.k.set_target(gain, num_samples);

        self.g.set_target(g / gain.sqrt(), num_samples);
    }

    /// like `Self::set_params_high_shelving` but smoothed.
    pub fn set_params_high_shelving_smoothed(
        &mut self,
        w_c: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize,
    ) {
        let g = Self::pre_gain_from_cutoff(w_c);

        self.k.set_target(gain, num_samples);

        self.g.set_target(g * gain.sqrt(), num_samples);
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

        let s = self.s.get_current();
        let g = self.g.get_current();

        self.lp = self.s.process(sample - s, g);
        self.hp = sample - self.lp;
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
