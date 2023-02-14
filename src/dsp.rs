pub mod sample;
pub mod graph;
pub mod processor;

/// 2 ^ (1/12)
const SEMITONE: f32 = 1.059_463_1;

/// linear interpolation on the unit interval
#[inline(always)]
pub fn lerp(y1: f32, y2: f32, a: f32) -> f32 { (y2 - y1).mul_add(a, y1) }

/// linear interpolation in a wave frame
/// # Safety
/// This function does not perform bounds checking,
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

/// semitones to frequency ratio
#[inline(always)]
pub fn semitones(x: f32) -> f32 {
    SEMITONE.powf(x)
}