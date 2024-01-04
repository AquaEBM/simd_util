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
pub struct OnePole<const N: usize = FLOATS_PER_VECTOR>
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
    fn set_values_smoothed(&mut self, g: Simd<f32, N>, k: Simd<f32, N>, inc: Simd<f32, N>) {
        self.g1.set_increment(Self::g1(g), inc);
        self.k.set_increment(k, inc);
    }

    /// like `Self::set_params` but smoothed
    #[inline]
    pub fn set_params_smoothed(
        &mut self,
        w_c: Simd<f32, N>,
        gain: Simd<f32, N>,
        inc: Simd<f32, N>,
    ) {
        self.set_values_smoothed(Self::g(w_c), gain, inc)
    }

    /// like `Self::set_params_low_shelving` but smoothed
    #[inline]
    pub fn set_params_low_shelving_smoothed(
        &mut self,
        w_c: Simd<f32, N>,
        gain: Simd<f32, N>,
        inc: Simd<f32, N>,
    ) {
        self.set_values_smoothed(Self::g(w_c) / gain.sqrt(), gain, inc)
    }

    /// like `Self::set_params_high_shelving` but smoothed.
    #[inline]
    pub fn set_params_high_shelving_smoothed(
        &mut self,
        w_c: Simd<f32, N>,
        gain: Simd<f32, N>,
        inc: Simd<f32, N>,
    ) {
        self.set_values_smoothed(Self::g(w_c) * gain.sqrt(), gain, inc)
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

        self.lp = self.s.tick((x - s) * g1);
        self.x = x;
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
    pub fn get_highpass(&self) -> Simd<f32, N> {
        self.x - self.lp
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
            LP => Self::get_lowpass,
            AP => Self::get_allpass,
            HP => Self::get_highpass,
            LSH => Self::get_low_shelf,
            HSH => Self::get_high_shelf,
        }
    }

    pub fn get_update_function(mode: FilterMode) -> fn(&mut Self, Simd<f32, N>, Simd<f32, N>) {
        use FilterMode::*;

        match mode {
            HSH => Self::set_params_high_shelving,
            LSH => Self::set_params_low_shelving,
            _ => Self::set_params,
        }
    }

    pub fn get_smoothing_update_function(
        mode: FilterMode,
    ) -> fn(&mut Self, Simd<f32, N>, Simd<f32, N>, Simd<f32, N>) {
        use FilterMode::*;

        match mode {
            LSH => Self::set_params_low_shelving_smoothed,
            HSH => Self::set_params_high_shelving_smoothed,
            _ => Self::set_params_smoothed,
        }
    }
}

#[cfg(feature = "transfer_funcs")]
impl<const _N: usize> OnePole<_N>
where
    LaneCount<_N>: SupportedLaneCount,
{
    pub fn get_transfer_function<T: Float>(
        filter_mode: FilterMode,
    ) -> fn(Complex<T>, T) -> Complex<T> {
        use FilterMode::*;

        match filter_mode {
            LP => Self::low_pass_impedance,
            AP => Self::all_pass_impedance,
            HP => Self::high_pass_impedance,
            LSH => Self::low_shelf_impedance,
            HSH => Self::high_shelf_impedance,
        }
    }

    fn h_denominator<T: Float>(s: Complex<T>) -> Complex<T> {
        s + T::one()
    }

    pub fn low_pass_impedance<T: Float>(s: Complex<T>, _gain: T) -> Complex<T> {
        Self::h_denominator(s).finv()
    }

    pub fn all_pass_impedance<T: Float>(s: Complex<T>, _gain: T) -> Complex<T> {
        (-s + T::one()).fdiv(Self::h_denominator(s))
    }

    pub fn high_pass_impedance<T: Float>(s: Complex<T>, _gain: T) -> Complex<T> {
        s.fdiv(Self::h_denominator(s))
    }

    pub fn low_shelf_impedance<T: Float>(s: Complex<T>, gain: T) -> Complex<T> {
        Self::tilting_impedance(s, gain.recip()).scale(gain.sqrt())
    }

    pub fn tilting_impedance<T: Float>(s: Complex<T>, gain: T) -> Complex<T> {
        let m = gain.sqrt();
        (s.scale(m) + T::one()) / (s + m)
    }

    pub fn high_shelf_impedance<T: Float>(s: Complex<T>, gain: T) -> Complex<T> {
        Self::tilting_impedance(s, gain).scale(gain.sqrt())
    }
}
