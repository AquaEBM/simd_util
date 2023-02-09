use std::{ops::{Deref, DerefMut}, borrow::Borrow};
use crate::util::{find_remove, permute};

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

#[derive(Debug)]
pub struct AudioGraph<I, D> {
    ordered_nodes: Vec<AudioGraphNode<I, D>>,
}

impl<I, D> Default for AudioGraph<I, D> {
    fn default() -> Self {
        Self { ordered_nodes: vec![] }
    }
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