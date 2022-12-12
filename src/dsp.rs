use std::{ops::{DerefMut, Add}, iter::Sum};

pub struct StereoSample {
    pub l: f32,
    pub r: f32
}

impl Add for StereoSample {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            l: self.l + rhs.l,
            r: self.r + rhs.r
        }
    }
}

impl Sum for StereoSample {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self { l: 0., r: 0. }, Add::add)
    }
}

/// 2 ^ (1/12)
const SEMITONE: f32 = 1.0594630943592953;

pub trait Processor {

    fn num_voices(&self) -> usize;

    // we don't pass in normalized frequencies to minimize round-off error
    fn add_voice(&mut self, freq: f32, sr: f32);

    fn remove_voice(&mut self, voice_idx: usize);

    fn process(&mut self, voice_idx: usize, channel_idx: usize) -> f32;

    /// do not implement if `T` is a producer that doesn't take inputs
    fn input(&mut self, _sample: f32, _voice_idx: usize, _channel_idx: usize) {}

    fn get_sample(&mut self) -> StereoSample {
        (0..self.num_voices()).map( |voice_idx| StereoSample {
            l: self.process(voice_idx, 0),
            r: self.process(voice_idx, 1)
        }).sum()
    }
}

impl<T, U> Processor for U
where
    T: Processor,
    U: DerefMut<Target = [T]>
{
    fn add_voice(&mut self, freq: f32, sr: f32) {
        self.iter_mut().for_each(|proc| proc.add_voice(freq, sr));
    }

    fn remove_voice(&mut self, voice_idx: usize) {
        self.iter_mut().for_each(|proc| proc.remove_voice(voice_idx));
    }

    fn process(&mut self, voice_idx: usize, channel_idx: usize) -> f32 {
        self.iter_mut().map(|proc| proc.process(voice_idx, channel_idx)).sum()
    }

    fn input(&mut self, sample: f32, voice_idx: usize, channel_idx: usize) {
        self.iter_mut().for_each(|proc| proc.input(sample, voice_idx, channel_idx));
    }

    fn num_voices(&self) -> usize {
        // the way the other methods are implemented guarantees that each
        // member has the same number of voices so this is fine
        self[0].num_voices()
    }
}

/// linear interpolation on the unit interval
#[inline(always)]
pub fn lerp(y1: f32, y2: f32, a: f32) -> f32 { (y2 - y1).mul_add(a, y1) }

/// linear interpolation in a wave frame: This function does not perform bounds checking,
/// the caller must. therefore, ensure that `0 <= x <`**`table.len() - 1`**
#[inline(always)]
pub unsafe fn lerp_table(table: &[f32], x: f32) -> f32 {
    let i: usize = x.to_int_unchecked();
    lerp(
        *table.get_unchecked(i),
        *table.get_unchecked(i + 1),
        x.fract()
    )
}

/// bilinear interpolation in a wavetable, same safety conditions as lerp_table.
#[inline]
pub unsafe fn lerp_terrain(terrain: &[impl AsRef<[f32]>], a: f32, b: f32) -> f32 {
    let i: usize = b.to_int_unchecked();

    lerp(
        lerp_table(terrain.get_unchecked(i).as_ref(), a),
        lerp_table(terrain.get_unchecked(i + 1).as_ref(), a),
        b.fract()
    )
}

#[inline]
// from http://paulbourke.net/miscellaneous/interpolation/
pub fn cubic_interp(y0: f32, y1: f32, y2: f32, y3: f32, mu: f32) -> f32 {
    let a = y3 - y2 - y0 + y1;
    let b = y0 - y1 - a;
    let c = y2 - y0;

    let mu2 = mu * mu;
    a * mu * mu2 + b * mu2 * c * mu + y1
}

/// semitones to frequency ratio
#[inline(always)]
pub fn semitones(x: f32) -> f32 {
    SEMITONE.powf(x)
}

/// tanh soft clipping: sample: [-1 ; 1], drive : [0 ; +inf]
#[inline]
pub fn soft_clip(sample: f32, drive: f32) -> f32 {
    (sample * drive).tanh()
}

/// hard clip a sample by clamping it in the range
/// [-drive, drive] without affecting its original magnitude
#[inline]
pub fn hard_clip(sample: f32, drive: f32) -> f32 {
    sample.clamp(-drive, drive)
}

#[inline]
pub fn bit_crush(sample: f32, drive: f32) -> f32 {
    (sample / drive).trunc() * drive
}