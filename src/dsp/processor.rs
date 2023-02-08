use arrayvec::ArrayVec;
use super::sample::*;

pub trait Processor: Send {

    fn add_voice(&mut self, norm_freq: f32);

    fn remove_voice(&mut self, voice_idx: usize);

    fn process(&mut self, inputs: &mut [StereoSample]);
}

#[derive(Default)]
pub struct ProcessSchedule { 
    nodes: Vec<ProcessComponent>,
    edges: Vec<Vec<usize>>,
}

impl Processor for ProcessSchedule {

    fn add_voice(&mut self, norm_freq: f32) {

        for node in &mut self.nodes {
            node.sample_buffer.push(ZERO_SAMPLE);
            node.processor.add_voice(norm_freq);
        }
    }

    fn remove_voice(&mut self, voice_idx: usize) {

        for node in &mut self.nodes {
            node.sample_buffer.swap_remove(voice_idx);
            node.processor.remove_voice(voice_idx);
        }
    }

    fn process(&mut self, inputs: &mut [StereoSample]) {

        // C++ like index iteration is required here in order to work around Rust's borrowing
        // rules because indexing, as opposed to, say, iter_mut() doesn't hold a long borrow

        for i in 0..self.nodes.len() {

            self.nodes[i].process();

            if  self.edges[i].is_empty() {

                self.nodes[i].output_to_buffer(inputs);
            }

            for &j in &self.edges[i] {

                for k in 0..self.nodes[i].sample_buffer.len() {

                    let sample = self.nodes[i].sample_buffer[k];

                    self.nodes[j].sample_buffer[k] += sample;
                }
            }
        }
    }
}

impl ProcessSchedule {
    pub fn push(&mut self, processor: Box<dyn Processor>, successors: Vec<usize>) {
        self.nodes.push(processor.into());
        self.edges.push(successors);
    }
}

pub struct ProcessComponent {
    processor: Box<dyn Processor>,
    sample_buffer: ArrayVec<StereoSample, 16>,
}

impl From<Box<dyn Processor>> for ProcessComponent {
    fn from(processor: Box<dyn Processor>) -> Self {
        Self {
            processor,
            sample_buffer: Default::default()
        }
    }
}

impl ProcessComponent {

    pub fn process(&mut self) {
        self.processor.process(&mut self.sample_buffer);
    }

    pub fn output_to_buffer(&mut self, inputs: &mut [StereoSample]) {

        for (&output, input) in self.sample_buffer.iter().zip(inputs.iter_mut()) {
            *input += output;
        }
    }
}