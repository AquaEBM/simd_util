use arrayvec::ArrayVec;
use super::sample::*;

pub trait Processor: Send {

    fn add_voice(&mut self, norm_freq: f32);

    fn remove_voice(&mut self, voice_idx: usize);

    fn process(&mut self, inputs: &mut [StereoSample]);
}

#[derive(Default)]
pub struct AudioGraph {
    nodes: Vec<AudioGraphNode>,
    edges: Vec<Vec<usize>>,
}

impl Processor for AudioGraph {

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

impl AudioGraph {
    pub fn push(&mut self, processor: Box<dyn Processor>, successors: Vec<usize>) {
        self.nodes.push(processor.into());
        self.edges.push(successors);
    }
}

pub struct AudioGraphNode {
    processor: Box<dyn Processor>,
    sample_buffer: ArrayVec<StereoSample, 16>,
}

impl From<Box<dyn Processor>> for AudioGraphNode {
    fn from(processor: Box<dyn Processor>) -> Self {
        Self {
            processor,
            sample_buffer: Default::default()
        }
    }
}

impl AudioGraphNode {

    pub fn process(&mut self) {
        self.processor.process(&mut self.sample_buffer);
    }

    pub fn output_to_buffer(&mut self, inputs: &mut [StereoSample]) {

        for (&output, input) in self.sample_buffer.iter().zip(inputs.iter_mut()) {
            *input += output;
        }
    }
}

pub fn find_remove<T: Eq>(vec: &mut Vec<T>, object: &T) {
    let pos = vec.iter().position(|e| e == object).unwrap();
    vec.remove(pos);
}

pub fn has_duplicates<T: Eq + Clone>(slice: &[T]) -> bool {

    let mut visited = Vec::new();
    
    slice.iter().any(|e| {
        let dup = visited.contains(e);
        visited.push(e.clone());
        dup
    })
}

pub fn permute<T>(slice: &mut [T], indices: &mut [usize]) {

    assert_eq!(slice.len(), indices.len(), "slices must have the same length");
    assert!(!has_duplicates(indices), "indices must not have duplicates");
    assert!(indices.iter().all(|&i| i < indices.len()), "all indices must be valid");

    for i in 0..indices.len() {

        let mut current = i;

        while i != indices[current] {

            let next = indices[current];
            slice.swap(current, next);

            indices[current] = current;
            current = next;
        }

        indices[current] = current;
    }
}

fn position(edges: &[Edge], index: &usize) -> Option<usize> {
    edges.iter().position(|(Edge::Normal(i) | Edge::Feedback(i))| i == index)
}

fn contains(edges: &[Edge], index: &usize) -> bool {
    position(edges, index).is_some()    
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Edge {
    Normal(usize),
    Feedback(usize),
}

impl Edge {

    pub fn set_as_feedback(&mut self) {
        let Self::Normal(i) = self else {
            panic!("edge is already a feedback edge");
        };

        *self = Self::Feedback(*i);
    }
}

#[derive(Debug)]
pub struct AudioGraphNode<I, D> {
    pub data: D,
    id: I,
    edges: Vec<Edge>,
}

impl<I, D> AudioGraphNode<I, D> {
    pub fn new(id: I, data: D) -> Self { Self { data, id, edges: vec![] } }
    pub fn id(&self) -> &I { &self.id }
    pub fn edges(&self) -> &[Edge] { &self.edges }
}

#[derive(Default, Debug)]
pub struct AudioGraph<I, D> {
    ordered_nodes: Vec<AudioGraphNode<I, D>>,
}

impl<I, D> Deref for AudioGraph<I, D> {
    type Target = [AudioGraphNode<I, D>];

    fn deref(&self) -> &Self::Target {
        &self.ordered_nodes
    }
}

impl<I, D> DerefMut for AudioGraph<I, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ordered_nodes
    }
}

impl<I, D> AudioGraph<I, D> {

    fn edges_mut(&mut self) -> impl Iterator<Item = &mut usize> {
        self.ordered_nodes.iter_mut()
            .flat_map(|node| node.edges.iter_mut())
            .map(|(Edge::Normal(i) | Edge::Feedback(i))| i)
    }

    pub fn top_level_insert(&mut self, id: I, data: D) {
        self.edges_mut().for_each(|edge| *edge += 1);
        self.ordered_nodes.insert(0, AudioGraphNode::new(id, data));
    }

    fn connect_indexes(&mut self, from: usize, to: usize) {
        if !contains(self[from].edges(), &to) {
            self[from].edges.push(Edge::Normal(to));
        }
    }

    /// Implementation of graph topological sort using Kahn's Algorithm
    fn topological_sort(&self) -> Option<Vec<usize>> {

        let mut outgoing_edges = Vec::from_iter(
            self.ordered_nodes.iter().map(|node| Vec::from_iter(
                node.edges().iter().filter_map(|edge| match edge {
                    Edge::Normal(i) => Some(*i),
                    _ => None,
                })
            ))
        );

        let mut incoming_edges = vec![vec![] ; self.ordered_nodes.len()];
        for (node, node_edges) in outgoing_edges.iter().enumerate() {
            for &edge in node_edges {
                incoming_edges[edge].push(node);
            }
        }

        let mut independent_nodes = Vec::from_iter(
            incoming_edges.iter()
                .enumerate()
                .filter_map(
                    |(i, edges)| {
                        edges.is_empty().then_some(i)
                    }
                )
        );

        let mut new_ordering = Vec::with_capacity(self.ordered_nodes.len());

        while let Some(node) = independent_nodes.pop() {

            new_ordering.push(node);

            while let Some(next_node) = outgoing_edges[node].pop() {

                let edges = &mut incoming_edges[next_node];

                find_remove(edges, &node);

                if edges.is_empty() {
                    independent_nodes.push(next_node);
                }
            }
        }

        incoming_edges.iter().all(Vec::is_empty).then_some(new_ordering)
    }

    fn reorder(&mut self, indices: &mut [usize]) {

        self.edges_mut().for_each(
            |edge| *edge = indices.iter().position(|i| i == edge).unwrap()
        );

        permute(self, indices);
    }

    pub fn find_node<Q>(&mut self, node_id: &Q) -> usize
    where
        I: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.iter().position(|node| node.id.borrow() == node_id).unwrap()
    }

    pub fn connect<Q>(&mut self, from_id: &Q, to_id: &Q)
    where
        I: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let from_index = self.find_node(from_id);
        let to_index = self.find_node(to_id);

        self.connect_indexes(from_index, to_index);

        if let Some(mut indices) = self.topological_sort() {

            self.reorder(&mut indices);

        } else {
            self[from_index].edges.last_mut().unwrap().set_as_feedback();
        }
    }
}