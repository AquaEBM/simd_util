use super::{simd_util::map, smoothing::*, *};

#[cfg_attr(feature = "nih_plug", derive(Enum))]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default, PartialOrd, Ord, Hash)]
pub enum FilterMode {
    #[cfg_attr(feature = "nih_plug", name = "Lowpass")]
    LP,
    #[cfg_attr(feature = "nih_plug", name = "Bandpass")]
    BP,
    #[cfg_attr(feature = "nih_plug", name = "Unit Bandpass")]
    BP1,
    #[cfg_attr(feature = "nih_plug", name = "Highpass")]
    HP,
    #[cfg_attr(feature = "nih_plug", name = "Allpass")]
    #[default]
    AP,
    #[cfg_attr(feature = "nih_plug", name = "Notch")]
    NCH,
    #[cfg_attr(feature = "nih_plug", name = "Low shelf")]
    LSH,
    #[cfg_attr(feature = "nih_plug", name = "Band shelf")]
    BSH,
    #[cfg_attr(feature = "nih_plug", name = "High Shelf")]
    HSH,
}

/// Digital implementation of the analogue SVF Filter, with built-in
/// parameter smoothing. Based on the one in the book The Art of VA
/// Filter Design by Vadim Zavalishin
///
/// Capable of outputing many different filter types,
/// (highpass, lowpass, bandpass, notch, shelving....)
#[derive(Default)]
pub struct SVF<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    g: LogSmoother<N>,
    r: LogSmoother<N>,
    k: LogSmoother<N>,
    s: [Integrator<N>; 2],
    x: Simd<f32, N>,
    hp: Simd<f32, N>,
    bp: Simd<f32, N>,
    lp: Simd<f32, N>,
}

impl<const N: usize> SVF<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub fn reset(&mut self) {
        self.s.iter_mut().for_each(Integrator::reset)
    }

    #[inline]
    fn g(w_c: Simd<f32, N>) -> Simd<f32, N> {
        // TODO: use a tan approximation to leverage SIMD
        // instead of calling scalar tan on each lane
        map(w_c * Simd::splat(0.5), f32::tan)
    }

    #[inline]
    fn set_values(&mut self, g: Simd<f32, N>, res: Simd<f32, N>, gain: Simd<f32, N>) {
        self.k.set_instantly(gain);
        self.g.set_instantly(g);
        self.r.set_instantly(res);
    }

    /// call this if you intend to later output _only_ low-shelving filter shapes
    #[inline]
    pub fn set_params_low_shelving(
        &mut self,
        w_c: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>,
    ) {
        let m2 = gain.sqrt();
        let g = Self::g(w_c);
        self.set_values(g / m2.sqrt(), res, m2);
    }

    /// call this if you intend to later output _only_ band-shelving filter shapes
    #[inline]
    pub fn set_params_band_shelving(
        &mut self,
        w_c: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>,
    ) {
        let g = Self::g(w_c);
        self.set_values(g, res / gain.sqrt(), gain);
    }

    /// call this if you intend to later output _only_ high-shelving filter shapes
    #[inline]
    pub fn set_params_high_shelving(
        &mut self,
        w_c: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>,
    ) {
        let m2 = gain.sqrt();
        let g = Self::g(w_c);
        self.set_values(g * m2.sqrt(), res, m2);
    }

    /// call this if you intend to later output non-shelving filter shapes
    #[inline]
    pub fn set_params(&mut self, w_c: Simd<f32, N>, res: Simd<f32, N>, _gain: Simd<f32, N>) {
        self.set_values(Self::g(w_c), res, Simd::splat(1.));
    }

    #[inline]
    fn set_values_smoothed(
        &mut self,
        g: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize,
    ) {
        self.k.set_target(gain, num_samples);
        self.g.set_target(g, num_samples);
        self.r.set_target(res, num_samples);
    }

    /// Like `Self::set_params_low_shelving` but with smoothing
    #[inline]
    pub fn set_params_low_shelving_smoothed(
        &mut self,
        w_c: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize,
    ) {
        let m2 = gain.sqrt();
        let g = Self::g(w_c);
        self.set_values_smoothed(g / m2.sqrt(), res, m2, num_samples);
    }

    /// Like `Self::set_params_band_shelving` but with smoothing
    #[inline]
    pub fn set_params_band_shelving_smoothed(
        &mut self,
        w_c: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize,
    ) {
        let g = Self::g(w_c);
        self.set_values_smoothed(g, res / gain.sqrt(), gain, num_samples);
    }

    /// Like `Self::set_params_high_shelving` but with smoothing
    #[inline]
    pub fn set_params_high_shelving_smoothed(
        &mut self,
        w_c: Simd<f32, N>,
        res: Simd<f32, N>,
        gain: Simd<f32, N>,
        num_samples: usize,
    ) {
        let m2 = gain.sqrt();
        let g = Self::g(w_c);
        self.set_values_smoothed(g * m2.sqrt(), res, m2, num_samples);
    }

    /// Like `Self::set_params_non_shelving` but with smoothing
    #[inline]
    pub fn set_params_smoothed(
        &mut self,
        w_c: Simd<f32, N>,
        res: Simd<f32, N>,
        _gain: Simd<f32, N>,
        num_samples: usize,
    ) {
        self.g.set_target(Self::g(w_c), num_samples);
        self.r.set_target(res, num_samples);
        self.k.set_instantly(Simd::splat(1.));
    }

    /// Update the filter's internal parameter smoothers.
    ///
    /// After calling `Self::set_params_<output_type>_smoothed(values, ..., num_samples)` this
    /// function should be called _up to_ `num_samples` times, until, that function is to be
    /// called again, calling this function more than `num_samples` times might result in
    /// the internal parameter states diverging away from the previously set values
    #[inline]
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
    #[inline]
    pub fn process(&mut self, sample: Simd<f32, N>) {

        let g = self.g.get_current();
        let s1 = self.s[0].get_current();
        let s2 = self.s[1].get_current();

        let g1 = self.r.get_current() + g;

        self.hp = g1.mul_add(-s1, sample - s2) / g1.mul_add(g, Simd::splat(1.));

        self.bp = self.s[0].process(self.hp * g);
        self.lp = self.s[1].process(self.bp * g);
        self.x = sample;
    }

    #[inline]
    fn get_gain(&self) -> Simd<f32, N> {
        self.k.get_current()
    }

    #[inline]
    pub fn get_lowpass(&self) -> Simd<f32, N> {
        self.lp
    }

    #[inline]
    pub fn get_bandpass(&self) -> Simd<f32, N> {
        self.bp
    }

    #[inline]
    pub fn get_unit_bandpass(&self) -> Simd<f32, N> {
        self.r.get_current() * self.bp
    }

    #[inline]
    pub fn get_highpass(&self) -> Simd<f32, N> {
        self.hp
    }

    #[inline]
    pub fn get_allpass(&self) -> Simd<f32, N> {
        // 2 * bp1 - x
        self.r.get_current().mul_add(self.bp + self.bp, -self.x)
    }

    #[inline]
    pub fn get_notch(&self) -> Simd<f32, N> {
        // x - bp1
        self.bp.mul_add(-self.r.get_current(), self.x)
    }

    #[inline]
    pub fn get_high_shelf(&self) -> Simd<f32, N> {
        let m2 = self.get_gain();
        let bp1 = self.get_unit_bandpass();
        m2.mul_add(m2.mul_add(self.hp, bp1), self.lp)
    }

    #[inline]
    pub fn get_band_shelf(&self) -> Simd<f32, N> {
        let bp1 = self.get_unit_bandpass();
        bp1.mul_add(self.get_gain(), self.x - bp1)
    }

    #[inline]
    pub fn get_low_shelf(&self) -> Simd<f32, N> {
        let m2 = self.get_gain();
        let bp1 = self.get_unit_bandpass();
        m2.mul_add(m2.mul_add(self.lp, bp1), self.hp)
    }

    pub fn get_output_function(mode: FilterMode) -> fn(&Self) -> Simd<f32, N> {
        use FilterMode::*;

        match mode {
            LP => Self::get_lowpass,
            BP => Self::get_bandpass,
            BP1 => Self::get_unit_bandpass,
            HP => Self::get_highpass,
            AP => Self::get_allpass,
            NCH => Self::get_notch,
            LSH => Self::get_low_shelf,
            BSH => Self::get_band_shelf,
            HSH => Self::get_high_shelf,
        }
    }

    pub fn get_update_function(
        mode: FilterMode,
    ) -> fn(&mut Self, Simd<f32, N>, Simd<f32, N>, Simd<f32, N>) {
        use FilterMode::*;

        match mode {
            LSH => Self::set_params_low_shelving,
            BSH => Self::set_params_band_shelving,
            HSH => Self::set_params_high_shelving,
            _ => Self::set_params,
        }
    }

    pub fn get_smoothing_update_function(
        mode: FilterMode,
    ) -> fn(&mut Self, Simd<f32, N>, Simd<f32, N>, Simd<f32, N>, usize) {
        use FilterMode::*;

        match mode {
            LSH => Self::set_params_low_shelving_smoothed,
            BSH => Self::set_params_band_shelving_smoothed,
            HSH => Self::set_params_high_shelving_smoothed,
            _ => Self::set_params_smoothed,
        }
    }
}

#[cfg(feature = "transfer_funcs")]
use ::num::{Float, Complex, One};
#[cfg(feature = "transfer_funcs")]
impl<const _N: usize> SVF<_N>
where
    LaneCount<_N>: SupportedLaneCount,
{
    pub fn get_transfer_function<T: Float>(
        filter_mode: FilterMode
    ) -> fn(Complex<T>, T, T) -> Complex<T> {
        
        use FilterMode::*;

        match filter_mode {
            LP => Self::low_pass_impedance,
            BP => Self::band_pass_impedance,
            BP1 => Self::unit_band_pass_impedance,
            HP => Self::high_pass_impedance,
            AP => Self::all_pass_impedance,
            NCH => Self::notch_impedance,
            LSH => Self::low_shelf_impedance,
            BSH => Self::band_shelf_impedance,
            HSH => Self::high_shelf_impedance,
        }
    }

    fn two<T: Float>(res: T) -> T {
        T::from(2f32).unwrap() * res
    }

    fn h_denominator<T: Float>(s: Complex<T>, res: T) -> Complex<T> {
        s * (s + Self::two(res)) + T::one()
    }

    pub fn low_pass_impedance<T: Float>(s: Complex<T>, res: T, _gain: T) -> Complex<T> {
        Self::h_denominator(s, res).finv()
    }

    pub fn band_pass_impedance<T: Float>(s: Complex<T>, res: T, _gain: T) -> Complex<T> {
        s.fdiv(Self::h_denominator(s, res))
    }

    pub fn unit_band_pass_impedance<T: Float>(s: Complex<T>, res: T, _gain: T) -> Complex<T> {
        Self::band_pass_impedance(s, res, _gain).scale(Self::two(res))
    }

    pub fn high_pass_impedance<T: Float>(s: Complex<T>, res: T, _gain: T) -> Complex<T> {
        (s * s).fdiv(Self::h_denominator(s, res))
    }

    pub fn all_pass_impedance<T: Float>(s: Complex<T>, res: T, _gain: T) -> Complex<T> {
        let bp1 = Self::unit_band_pass_impedance(s, res, _gain);
        bp1 + bp1 - Complex::one()
    }

    pub fn notch_impedance<T: Float>(s: Complex<T>, res: T, _gain: T) -> Complex<T> {
        Complex::<T>::one() - Self::unit_band_pass_impedance(s, res, _gain)
    }

    pub fn tilting_impedance<T: Float>(s: Complex<T>, res: T, gain: T) -> Complex<T> {
        let m2 = gain.sqrt();
        let m = m2.sqrt();
        let sm = s.unscale(m);
        (s * s + sm.scale(Self::two(res)) + m2.recip()).fdiv(Self::h_denominator(sm, res))
    }

    pub fn low_shelf_impedance<T: Float>(s: Complex<T>, res: T, gain: T) -> Complex<T> {
        let m2 = gain.sqrt();
        Self::tilting_impedance(s, res, gain.recip()).scale(m2)
    }

    pub fn band_shelf_impedance<T: Float>(s: Complex<T>, res: T, gain: T) -> Complex<T> {
        let m = gain.sqrt();
        (s * (s + Self::two(res) * m) + T::one()).fdiv(Self::h_denominator(s, res / m))
    }

    pub fn high_shelf_impedance<T: Float>(s: Complex<T>, res: T, gain: T) -> Complex<T> {
        let m2 = gain.sqrt();
        Self::tilting_impedance(s, res, gain).scale(m2)
    }
}