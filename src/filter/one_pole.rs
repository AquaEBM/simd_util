use super::{simd_util::map, smoothing::*, *};

#[derive(Default)]
pub struct OnePole<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    g1: LinearSmoother<N>,
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
        self.s.reset()
    }

    fn g(w_c: Simd<f32, N>) -> Simd<f32, N> {
        map(w_c * Simd::splat(0.5), f32::tan)
    }

    fn g1(g: Simd<f32, N>) -> Simd<f32, N> {
        g / (Simd::splat(1.) + g)
    }

    fn set_values(&mut self, g: Simd<f32, N>, k: Simd<f32, N>) {
        self.g1.set_instantly(Self::g1(g));
        self.k.set_instantly(k);
    }

    /// call this _only_ if you intend to
    /// output non-shelving filter shapes.
    pub fn set_params(&mut self, w_c: Simd<f32, N>, gain: Simd<f32, N>) {
        self.set_values(Self::g(w_c), gain)
    }

    /// call this _only_ if you intend to output low-shelving filter shapes.
    pub fn set_params_low_shelving(&mut self, w_c: Simd<f32, N>, gain: Simd<f32, N>) {
        self.k.set_instantly(gain);
        self.g1.set_instantly(Self::g(w_c) / gain.sqrt());
    }

    /// call this _only_ if you intend to output high-shelving filter shapes.
    pub fn set_params_high_shelving(&mut self, w_c: Simd<f32, N>, gain: Simd<f32, N>) {
        self.k.set_instantly(gain);
        self.g1.set_instantly(Self::g(w_c) * gain.sqrt());
    }

    fn set_values_smoothed(&mut self, g: Simd<f32, N>, k: Simd<f32, N>, num_samples: usize) {
        self.g1.set_target(Self::g1(g), num_samples);
        self.k.set_target(k, num_samples);
    }

    /// like `Self::set_params` but smoothed
    pub fn set_params_smoothed(
        &mut self,
        w_c: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize,
    ) {
        self.set_values_smoothed(Self::g(w_c), gain, num_samples)
    }

    /// like `Self::set_params_low_shelving` but smoothed
    pub fn set_params_low_shelving_smoothed(
        &mut self,
        w_c: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize,
    ) {
        self.set_values_smoothed(
            Self::g(w_c) / gain.sqrt(),
            gain,
            num_samples,
        )
    }

    /// like `Self::set_params_high_shelving` but smoothed.
    pub fn set_params_high_shelving_smoothed(
        &mut self,
        w_c: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize,
    ) {
        self.set_values_smoothed(
            Self::g(w_c) * gain.sqrt(),
            gain,
            num_samples,
        )
    }

    /// update the filter's internal parameter smoothers.
    ///
    /// After calling `Self::set_params_smoothed([values, ...], num_samples)` this should
    /// be called only _once_ per sample, _up to_ `num_samples` times, until
    /// `Self::set_params_smoothed` is to be called again
    pub fn update_smoothers(&mut self) {
        self.g1.tick();
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
        let g1 = self.g1.get_current();

        self.lp = self.s.process(sample - s, g1);
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
