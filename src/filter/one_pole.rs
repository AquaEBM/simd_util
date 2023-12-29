use super::{smoothing::*, *};

#[cfg_attr(feature = "nih_plug", derive(Enum))]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default, PartialOrd, Ord, Hash)]
pub enum FilterMode {
    #[cfg_attr(feature = "nih_plug", name = "Highpass")]
    HP,
    #[cfg_attr(feature = "nih_plug", name = "Lowpass")]
    LP,
    #[cfg_attr(feature = "nih_plug", name = "Allpass")]
    #[default]
    AP,
    #[cfg_attr(feature = "nih_plug", name = "Low Shelf")]
    LSH,
    #[cfg_attr(feature = "nih_plug", name = "High Shelf")]
    HSH,
}

#[derive(Default)]
pub struct OnePole<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    g1: LogSmoother<N>,
    k: LogSmoother<N>,
    s: Integrator<N>,
    lp: Simd<f32, N>,
    x: Simd<f32, N>,
}

impl<const N: usize> OnePole<N>
where
    LaneCount<N>: SupportedLaneCount,
{   
    #[inline]
    pub fn reset(&mut self) {
        self.s.reset()
    }

    #[inline]
    fn g(w_c: Simd<f32, N>) -> Simd<f32, N> {
        math::tan_half_x(w_c)
    }

    #[inline]
    fn g1(g: Simd<f32, N>) -> Simd<f32, N> {
        g / (Simd::splat(1.) + g)
    }

    #[inline]
    fn set_values(&mut self, g: Simd<f32, N>, k: Simd<f32, N>) {
        self.g1.set_instantly(Self::g1(g));
        self.k.set_instantly(k);
    }

    /// call this _only_ if you intend to
    /// output non-shelving filter shapes.
    #[inline]
    pub fn set_params(&mut self, w_c: Simd<f32, N>, gain: Simd<f32, N>) {
        self.set_values(Self::g(w_c), gain)
    }

    /// call this _only_ if you intend to output low-shelving filter shapes.
    #[inline]
    pub fn set_params_low_shelving(&mut self, w_c: Simd<f32, N>, gain: Simd<f32, N>) {
        self.k.set_instantly(gain);
        self.g1.set_instantly(Self::g(w_c) / gain.sqrt());
    }

    /// call this _only_ if you intend to output high-shelving filter shapes.
    #[inline]
    pub fn set_params_high_shelving(&mut self, w_c: Simd<f32, N>, gain: Simd<f32, N>) {
        self.k.set_instantly(gain);
        self.g1.set_instantly(Self::g(w_c) * gain.sqrt());
    }

    #[inline]
    fn set_values_smoothed(&mut self, g: Simd<f32, N>, k: Simd<f32, N>, num_samples: usize) {
        self.g1.set_target(Self::g1(g), num_samples);
        self.k.set_target(k, num_samples);
    }

    /// like `Self::set_params` but smoothed
    #[inline]
    pub fn set_params_smoothed(
        &mut self,
        w_c: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize,
    ) {
        self.set_values_smoothed(Self::g(w_c), gain, num_samples)
    }

    /// like `Self::set_params_low_shelving` but smoothed
    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn process(&mut self, x: Simd<f32, N>) {
        let s = self.s.get_current();
        let g1 = self.g1.get_current();

        self.lp = self.s.process((x - s) * g1);
        self.x = x;
    }

    #[inline]
    pub fn get_highpass(&self) -> Simd<f32, N> {
        self.x - self.lp
    }

    #[inline]
    pub fn get_lowpass(&self) -> Simd<f32, N> {
        self.lp
    }

    #[inline]
    pub fn get_allpass(&self) -> Simd<f32, N> {
        self.lp - self.get_highpass()
    }

    #[inline]
    pub fn get_low_shelf(&self) -> Simd<f32, N> {
        self.k.get_current() * self.lp + self.get_highpass()
    }

    #[inline]
    pub fn get_high_shelf(&self) -> Simd<f32, N> {
        self.k.get_current().mul_add(self.get_highpass(), self.lp)
    }

    pub fn get_output_function(mode: FilterMode) -> fn(&Self) -> Simd<f32, N> {
        use FilterMode::*;

        match mode {
            HP => Self::get_highpass,
            LP => Self::get_lowpass,
            AP => Self::get_allpass,
            HSH => Self::get_high_shelf,
            LSH => Self::get_low_shelf,
        }
    }

    pub fn get_update_function(
        mode: FilterMode,
    ) -> fn(&mut Self, Simd<f32, N>, Simd<f32, N>) {
        use FilterMode::*;

        match mode {
            HSH => Self::set_params_high_shelving,
            LSH => Self::set_params_low_shelving,
            _ => Self::set_params,
        }
    }

    pub fn get_smoothing_update_function(
        mode: FilterMode,
    ) -> fn(&mut Self, Simd<f32, N>, Simd<f32, N>, usize) {
        use FilterMode::*;

        match mode {
            HSH => Self::set_params_high_shelving_smoothed,
            LSH => Self::set_params_low_shelving_smoothed,
            _ => Self::set_params_smoothed,
        }
    }
}
