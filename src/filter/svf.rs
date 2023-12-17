use super::{simd_util::map, smoothing::*, *};

/// Digital implementation of the analogue SVF Filter, with built-in
/// parameter smoothing. Based on the one in the book The Art of VA
/// Filter Design by Vadim Zavalinish
///
/// Capable of outputing many different filter types,
/// (highpass, lowpass, bandpass, notch, shelving....)
#[derive(Default)]
pub struct SVF<const N: usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    g: LogSmoother<N>,
    g2: LogSmoother<N>,
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

    fn g(w_c: Simd<f32, N>) -> Simd<f32, N> {
        map(w_c * Simd::splat(0.5), f32::tan)
    }

    fn g2(g: Simd<f32, N>, res: Simd<f32, N>) -> Simd<f32, N> {
        let one = Simd::splat(1.);

        one / g.mul_add(g + res, one)
    }

    fn set_values(&mut self, g: Simd<f32, N>, res: Simd<f32, N>, gain: Simd<f32, N>) {
        self.r.set_instantly(res);
        self.k.set_instantly(gain);
        self.g.set_instantly(g);
        self.g2.set_instantly(Self::g2(g, res));

    }

    /// call this if you intend to later output _only_ low-shelving filter shapes
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
    pub fn set_params(&mut self, w_c: Simd<f32, N>, res: Simd<f32, N>, _gain: Simd<f32, N>) {
        self.set_values(Self::g(w_c), res, Simd::splat(1.));
    }

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
    /// the internal parameter states diverging from the previously set values
    pub fn update_all_smoothers(&mut self) {
        self.k.tick();
        self.r.tick();
        self.g.tick();
        self.g2.tick();
    }

    /// Update the filter's internal state, given the provided input sample.
    ///
    /// This should be called _only once_ per sample, _every sample_
    ///
    /// After calling this, you can get different filter outputs
    /// using `Self::get_{highpass, bandpass, notch, ...}`
    pub fn process(&mut self, sample: Simd<f32, N>) {
        let g = self.g.get_current();
        let g1 = g + self.r.get_current();
        let g2 = self.g2.get_current();
        let s1 = self.s[0].get_current();
        let s2 = self.s[1].get_current();

        self.x = sample;
        self.hp = ((sample - s2) - s1 * g1) * g2;
        self.bp = self.s[0].process(self.hp, g);
        self.lp = self.s[1].process(self.bp, g);
    }

    /// Get the current, potentially smoothed value of the internal gain parameter
    pub fn get_gain(&self) -> Simd<f32, N> {
        self.k.get_current()
    }

    pub fn get_highpass(&self) -> Simd<f32, N> {
        self.hp
    }

    pub fn get_bandpass(&self) -> Simd<f32, N> {
        self.bp
    }

    pub fn get_lowpass(&self) -> Simd<f32, N> {
        self.lp
    }

    pub fn get_bandpass1(&self) -> Simd<f32, N> {
        self.r.get_current() * self.bp
    }

    pub fn get_allpass(&self) -> Simd<f32, N> {
        let bp1 = self.get_bandpass1();
        bp1 + bp1 - self.x
    }

    pub fn get_notch(&self) -> Simd<f32, N> {
        self.x - self.get_bandpass1()
    }

    pub fn get_high_shelf(&self) -> Simd<f32, N> {
        let m2 = self.get_gain();
        let bp1 = self.get_bandpass1();
        m2.mul_add(m2.mul_add(self.hp, bp1), self.lp)
    }

    pub fn get_band_shelf(&self) -> Simd<f32, N> {
        let bp1 = self.get_bandpass1();
        bp1.mul_add(self.get_gain(), self.x - bp1)
    }

    pub fn get_low_shelf(&self) -> Simd<f32, N> {
        let m2 = self.get_gain();
        let bp1 = self.get_bandpass1();
        m2.mul_add(m2.mul_add(self.lp, bp1), self.hp)
    }
}
