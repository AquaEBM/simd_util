use super::*;

use simd::{Mask, SimdElement, f32x2, simd_swizzle};

use core::{cell::Cell, mem};

#[cfg(any(target_feature = "avx512f", target_feature = "avx2"))]
use core::arch::x86_64::*;

pub const MAX_VECTOR_WIDTH: usize = {
    if cfg!(target_feature = "avx512f") {
        64
    } else if cfg!(target_feature = "avx") {
        32
    } else if cfg!(any(target_feature = "sse", target_feature = "neon")) {
        16
    } else {
        8
    }
};

pub const FLOATS_PER_VECTOR: usize = MAX_VECTOR_WIDTH / size_of::<f32>();

pub type VFloat<const N: usize = FLOATS_PER_VECTOR> = Simd<f32, N>;
pub type VUInt<const N: usize = FLOATS_PER_VECTOR> = Simd<u32, N>;
pub type TMask<const N: usize = FLOATS_PER_VECTOR> = Mask<i32, N>;

/// Convenience function on simd types when specialized functions aren't
/// available in the standard library, hoping autovectorization compiles this
/// into an simd instruction
#[inline]
pub fn map<T: SimdElement, U: SimdElement, const N: usize>(
    v: Simd<T, N>,
    f: impl FnMut(T) -> U,
) -> Simd<U, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    v.to_array().map(f).into()
}

/// Like `Simd::gather_select_unckecked` but with a pointer and using `u32` offsets
///
/// # Safety
///
/// The same requirements as `Simd::gather_select_unchecked`
// We're using intrinsics for now because u32 gathers aren't in core::simd (yet?)
#[inline]
pub unsafe fn gather_select_unchecked(
    pointer: *const f32,
    index: VUInt,
    enable: TMask,
    or: VFloat,
) -> VFloat {
    unsafe {
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
        return Simd::gather_select_unchecked(
            core::slice::from_raw_parts(pointer, 0),
            enable.cast(),
            index.cast(),
            or,
        );

        #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
        return _mm256_mask_i32gather_ps(
            or.into(),
            pointer,
            index.into(),
            mem::transmute(enable), // Why is this __m256, not __m256i? I don't know
            4,
        )
        .into();

        #[cfg(target_feature = "avx512f")]
        return _mm512_mask_i32gather_ps(
            or.into(),
            enable.to_bitmask() as __mmask16,
            index.into(),
            pointer.cast(),
            4,
        )
        .into();
    }
}

/// Like `Simd::gather_select_unchecked` but with a pointer, `u32` offsets and all offsets are enabled
///
/// # Safety
///
/// The same as `Simd::gather_select_unchecked`
#[inline]
pub unsafe fn gather_unchecked(pointer: *const f32, index: VUInt) -> VFloat {
    unsafe {
        #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
        return Simd::gather_select_unchecked(
            core::slice::from_raw_parts(pointer, 0),
            Mask::splat(true),
            index.cast(),
            VFloat::splat(0.),
        );

        #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
        return _mm256_i32gather_ps(pointer, index.into(), 4).into();

        #[cfg(target_feature = "avx512f")]
        return _mm512_i32gather_ps(index.into(), pointer.cast(), 4).into();
    }
}

#[inline]
pub fn sum_to_stereo_sample(x: VFloat) -> f32x2 {
    unsafe {
        #[cfg(any(target_feature = "sse", target_feature = "neon"))]
        let x = {
            let [l, r]: [Simd<f32, { FLOATS_PER_VECTOR / 2 }>; 2] = mem::transmute(x);
            l + r
        };

        #[cfg(target_feature = "avx")]
        let x = {
            let [l, r]: [Simd<f32, { FLOATS_PER_VECTOR / 4 }>; 2] = mem::transmute(x);
            l + r
        };

        #[cfg(target_feature = "avx512f")]
        let x = {
            let [l, r]: [Simd<f32, { FLOATS_PER_VECTOR / 8 }>; 2] = mem::transmute(x);
            l + r
        };

        x
    }
}

pub const STEREO_VOICES_PER_VECTOR: usize = FLOATS_PER_VECTOR / 2;

// Safety argument for the six following functions:
//  - both referenced types have the same size, more specifically, 2 * STEREO_VOICES_PER_VECTOR
// is always equal to FLOATS_PER_VECTOR, because it is always a multiple of 2
//  - the referred-to type of `vector(s)` has greater alignment than that of the return type
//  - the output reference's lifetime is the same as that of the input, so no unbounded lifetimes
//  - we are transmuting a vector to an array over the same scalar, so values are valid

#[inline]
pub fn split_stereo<T: SimdElement>(
    vector: &Simd<T, FLOATS_PER_VECTOR>,
) -> &[Simd<T, 2>; STEREO_VOICES_PER_VECTOR] {
    // SAFETY: see above
    unsafe { mem::transmute(vector) }
}

#[inline]
pub fn split_stereo_slice<T: SimdElement>(
    vectors: &[Simd<T, FLOATS_PER_VECTOR>],
) -> &[[Simd<T, 2>; STEREO_VOICES_PER_VECTOR]] {
    // SAFETY: see above
    unsafe { mem::transmute(vectors) }
}

#[inline]
pub fn split_stereo_mut<T: SimdElement>(
    vector: &mut Simd<T, FLOATS_PER_VECTOR>,
) -> &mut [Simd<T, 2>; STEREO_VOICES_PER_VECTOR] {
    // SAFETY: see above
    unsafe { mem::transmute(vector) }
}

#[inline]
pub fn split_stereo_slice_mut<T: SimdElement>(
    vectors: &mut [Simd<T, FLOATS_PER_VECTOR>],
) -> &mut [[Simd<T, 2>; STEREO_VOICES_PER_VECTOR]] {
    // SAFETY: see above
    unsafe { mem::transmute(vectors) }
}

#[inline]
pub fn split_stereo_cell<T: SimdElement>(
    vector: &Cell<Simd<T, FLOATS_PER_VECTOR>>,
) -> &Cell<[Simd<T, 2>; STEREO_VOICES_PER_VECTOR]> {
    // SAFETY: see above
    unsafe { mem::transmute(vector) }
}

#[inline]
pub fn split_stereo_cell_slice<T: SimdElement>(
    vectors: &[Cell<Simd<T, FLOATS_PER_VECTOR>>],
) -> &[[Cell<Simd<T, 2>>; STEREO_VOICES_PER_VECTOR]] {
    // SAFETY: see above
    unsafe { mem::transmute(vectors) }
}

#[inline]
pub fn splat_stereo<T: SimdElement>(pair: Simd<T, 2>) -> Simd<T, FLOATS_PER_VECTOR> {
    const ZERO_ONE: [usize; FLOATS_PER_VECTOR] = {
        let mut array = [0; FLOATS_PER_VECTOR];
        let mut i = 1;
        while i < FLOATS_PER_VECTOR {
            array[i] = 1;
            i += 2;
        }
        array
    };

    simd_swizzle!(pair, ZERO_ONE)
}

/// Return a vector where values at the even
/// indices are at the odd ones and vice-versa
#[inline]
pub fn swap_stereo<T: SimdElement>(v: Simd<T, FLOATS_PER_VECTOR>) -> Simd<T, FLOATS_PER_VECTOR> {
    const FLIP_PAIRS: [usize; FLOATS_PER_VECTOR] = {
        let mut array = [0; FLOATS_PER_VECTOR];

        let mut i = 0;
        while i < FLOATS_PER_VECTOR {
            array[i] = i ^ 1;
            i += 1;
        }
        array
    };

    simd_swizzle!(v, FLIP_PAIRS)
}

/// triangluar panning of a vector of stereo samples, given 0 <= pan <= 1
#[inline]
pub fn triangular_pan_weights(pan_norm: VFloat) -> VFloat {
    const SIGN_MASK: VFloat = {
        let mut array = [0.; FLOATS_PER_VECTOR];
        let mut i = 0;
        while i < FLOATS_PER_VECTOR {
            array[i] = -0.;
            i += 2;
        }
        Simd::from_array(array)
    };

    const ALT_ONE: VFloat = {
        let mut array = [0.; FLOATS_PER_VECTOR];
        let mut i = 0;
        while i < FLOATS_PER_VECTOR {
            array[i] = 1.;
            i += 2;
        }
        Simd::from_array(array)
    };

    VFloat::from_bits(pan_norm.to_bits() ^ SIGN_MASK.to_bits()) + ALT_ONE
}

#[inline]
pub fn splat_slot<T: SimdElement>(
    vector: &Simd<T, FLOATS_PER_VECTOR>,
    index: usize,
) -> Option<Simd<T, FLOATS_PER_VECTOR>> {
    let array = split_stereo(vector);

    array.get(index).copied().map(splat_stereo)
}

/// # Safety
///
/// `index < STEREO_VOICES_PER_VECTOR` must hold
#[inline]
pub unsafe fn splat_slot_unchecked<T: SimdElement>(
    vector: &Simd<T, FLOATS_PER_VECTOR>,
    index: usize,
) -> Simd<T, FLOATS_PER_VECTOR> {
    let stereo_samples = split_stereo(vector);
    splat_stereo(*unsafe { stereo_samples.get_unchecked(index) })
}
