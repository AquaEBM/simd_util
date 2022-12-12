pub mod dsp;
pub mod gui;

pub const fn ilog2(n: usize) -> usize {
    (usize::BITS - 1 - n.leading_zeros()) as usize
}

pub fn all_but_last<T>(slice: &[T]) -> &[T] {
    slice.split_last().unwrap().1
}

pub fn all_but_last_mut<T>(slice: &mut [T]) -> &mut [T] {
    slice.split_last_mut().unwrap().1
}