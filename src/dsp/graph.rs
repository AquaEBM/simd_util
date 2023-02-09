use std::{ops::{Deref, DerefMut}, borrow::Borrow};
use crate::util::{find_remove, permute};

fn position(edges: &[Edge], index: &usize) -> Option<usize> {
    edges.iter().position(|(Edge::Normal(i) | Edge::Feedback(i))| i == index)
}

fn contains(edges: &[Edge], index: &usize) -> bool {
    position(edges, index).is_some()    
}

/// Implementation of graph topological sort using Kahn's Algorithm
fn topological_sort(mut nodes: Vec<Vec<usize>>) -> Option<Vec<usize>> {

    let mut incoming_edges = vec![vec![] ; nodes.len()];
    for (node, node_edges) in nodes.iter().enumerate() {
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

    let mut new_ordering = Vec::with_capacity(nodes.len());

    while let Some(i) = independent_nodes.pop() {

        new_ordering.push(i);

        while let Some(next) = nodes[i].pop() {

            let edges = &mut incoming_edges[next];

            find_remove(edges, &i);

            if edges.is_empty() {
                independent_nodes.push(next);
            }
        }
    }

    incoming_edges.iter().all(Vec::is_empty).then_some(new_ordering)
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

    pub fn index_if_normal(&self) -> Option<usize> {
        match self {
            &Edge::Normal(i) => Some(i),
            _ => None,
        }
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

    fn try_connect_indexes(&mut self, from: usize, to: usize) -> bool {
        let no_duplicates = !contains(self[from].edges(), &to);
        if no_duplicates {
            self[from].edges.push(Edge::Normal(to));
        }

        no_duplicates
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

    pub fn connect<Q>(&mut self, from_id: &Q, to_id: &Q) -> Option<((usize, usize), Option<Vec<usize>>)>
    where
        I: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let from_index = self.find_node(from_id);
        let to_index = self.find_node(to_id);

        let mut result = if self.try_connect_indexes(from_index, to_index) {
           Some(((from_index, to_index), None))
        } else {
            return None;
        };

        if let Some(indices) = topological_sort(
            // all non-feedback edges
            self.ordered_nodes
                .iter()
                .map(|node| node.edges()
                    .iter()
                    .filter_map(Edge::index_if_normal)
                    .collect()
                ).collect()
        ) {

            self.reorder(&mut indices.clone());
            result.as_mut().unwrap().1 = Some(indices);

        } else {
            self[from_index].edges.last_mut().unwrap().set_as_feedback();
        }

        result
    }
}