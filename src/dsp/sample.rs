use std::{ops::{Add, AddAssign, Mul, Sub, Div, Rem}, iter::Sum};
use crate::*;

use super::semitones;

pub const ZERO_SAMPLE: StereoSample = StereoSample { l: 0., r: 0. };

#[derive(Clone, Copy)]
pub struct StereoSample {
    pub l: f32,
    pub r: f32
}

impl StereoSample {
    pub fn semitones(self) -> Self {
        Self { l: semitones(self.l), r: semitones(self.r) }
    }

    pub fn min(self, rhs: Self) -> Self {
        Self { l: self.l.min(rhs.l), r: self.r.min(rhs.r)}
    }

    pub fn sqrt(self) -> Self {
        Self { l: self.l.sqrt(), r: self.r.sqrt() }
    }

    pub const fn splat(l @ r: f32) -> Self {
        Self { l, r }
    }

    pub fn mul_rev(self, rhs: Self) -> Self {
        Self {
            l: mul(self.l, rhs.r),
            r: mul(self.r, rhs.l),
        }
    }
}

impl Default for StereoSample {
    fn default() -> Self {
        ZERO_SAMPLE
    }
}

impl Add for StereoSample {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            l: add(self.l, rhs.l),
            r: add(self.r, rhs.r)
        }
    }
}

impl Mul for StereoSample {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            l: mul(self.l, rhs.l),
            r: mul(self.r, rhs.r)
        }
    }
}

impl Mul<f32> for StereoSample {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            l: mul(self.l, rhs),
            r: mul(self.r, rhs)
        }
    }
}

impl Sub for StereoSample {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            l: sub(self.l, rhs.l),
            r: sub(self.r, rhs.r)
        }
    }
}

impl Div for StereoSample {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            l: div(self.l, rhs.l),
            r: div(self.r, rhs.r)
        }
    }
}

impl Rem for StereoSample {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        Self {
            l: rem(self.l, rhs.l),
            r: rem(self.r, rhs.r)
        }
    }
}

impl AddAssign for StereoSample {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sum for StereoSample {
    #[inline(always)]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(ZERO_SAMPLE, Add::add)
    }
}

impl From<[f32 ; 2]> for StereoSample {
    #[inline(always)]
    fn from([l, r]: [f32 ; 2]) -> Self {
        Self { l, r }
    }
}

impl From<f32> for StereoSample {
    #[inline(always)]
    fn from(v: f32) -> Self {
        Self { l: v, r: v }
    }
}