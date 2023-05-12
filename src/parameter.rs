use std::{sync::{atomic::Ordering, Arc}, array};
use atomic_float::AtomicF32;
use nih_plug::prelude::{Param, ParamSetter};
use nih_plug_egui::egui::Id;
use std::ops::Deref;

pub trait Parameter {

    fn get_normalized_value(&self) -> f32;
    fn set_normalized_value(&self, value: f32);
    fn preview_nomrmalized(&self, value: f32) -> f32;

    fn name(&self) -> &str { "" }
    fn norm_val_to_string(&self, norm_val: f32) -> String {
        norm_val.to_string()
    }
    fn id(&self) -> Option<Id> { None }
    fn default_normalized_value(&self) -> f32 { 0. }
    fn begin_automation(&self) {}
    fn end_automation(&self) {}
}

pub struct Modulable<T, const POLYPHONY: usize> {
    param: T,
    modulation_buffer: Arc<[[AtomicF32 ; 2] ; POLYPHONY]>
}

impl<T, const N: usize> From<T> for Modulable<T, N> {
    fn from(param: T) -> Self {

        // AtomicF32 is neither Copy nor Clone so we need to do this.
        let modulation_buffer = Arc::new(array::from_fn(|_| [
            AtomicF32::new(0.),
            AtomicF32::new(0.)
        ]));

        Self { param, modulation_buffer }
    }
}

impl<T, const N: usize> Deref for Modulable<T, N> {
    type Target = T;

    fn deref(&self) -> &Self::Target { &self.param }
}

impl<T: Param, const N: usize> Modulable<T, N> 
{
    pub fn get_value(&self, voice_idx: usize) -> [T::Plain ; 2] {

        let sample = unsafe { self.modulation_buffer.get_unchecked(voice_idx) };
        [
            self.preview_modulated(sample[0].load(Ordering::Relaxed)),
            self.preview_modulated(sample[1].load(Ordering::Relaxed)),
        ]
    }
}

pub struct ParamHandle<'a, T> {
    param: T,
    setter: &'a ParamSetter<'a>
}

impl<'a, T> From<(T, &'a ParamSetter<'a>)> for ParamHandle<'a, T> {
    fn from((param, setter): (T, &'a ParamSetter<'a>)) -> Self {
        Self { param, setter }
    }
}

impl<'a, T, U> Parameter for ParamHandle<'a, U>
where
    T: Param + 'a,
    U: Deref<Target = T>
{
    fn get_normalized_value(&self) -> f32 {
        self.param.unmodulated_normalized_value()
    }

    fn set_normalized_value(&self, value: f32) {
        self.setter.set_parameter_normalized(self.param.deref(), value);
    }

    fn preview_nomrmalized(&self, value: f32) -> f32 {
        self.param.preview_normalized(self.param.preview_plain(value))
    }

    fn name(&self) -> &str { self.param.name() }

    fn id(&self) -> Option<Id> {
        Some(Id::new(self.param.as_ptr()))
    }

    fn norm_val_to_string(&self, norm_val: f32) -> String {
        self.param.normalized_value_to_string(norm_val, true)
    }

    fn begin_automation(&self) {
        self.setter.begin_set_parameter(self.param.deref());
    }

    fn end_automation(&self) {
        self.setter.end_set_parameter(self.param.deref());
    }

    fn default_normalized_value(&self) -> f32 { self.param.default_normalized_value() }
}