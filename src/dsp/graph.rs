use std::{ops::{Deref, DerefMut}};
use crate::util::{find_remove, Permute};

/// Implementation of graph topological sort using Kahn's Algorithm
fn topological_sort(nodes: &[Vec<usize>]) -> Option<Box<[usize]>> {

    let mut nodes = nodes.to_vec();

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

    assert_eq!(new_ordering.len(), nodes.len());
    incoming_edges.iter().all(Vec::is_empty).then_some(new_ordering.into_boxed_slice())
}

#[derive(Debug)]
pub struct AudioGraph<D> {
    ordered_nodes: Vec<D>,
    edges: Vec<Vec<usize>>
}

impl<D> AudioGraph<D> {
    pub fn edges(&self) -> &[Vec<usize>] { &self.edges }
}

impl<D> Default for AudioGraph<D> {
    fn default() -> Self {
        Self { ordered_nodes: vec![], edges: vec![] }
    }
}

impl<D> Deref for AudioGraph<D> {
    type Target = [D];

    fn deref(&self) -> &Self::Target { &self.ordered_nodes }
}

impl<D> DerefMut for AudioGraph<D> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.ordered_nodes }
}

impl<D> AudioGraph<D> {

    fn edges_mut(&mut self) -> impl Iterator<Item = &mut usize> {
        self.edges.iter_mut().flat_map(|edges| edges.iter_mut())
    }

    pub fn top_level_insert(&mut self, data: D) {
        self.ordered_nodes.push(data);
    }

    fn try_connect_indexes(&mut self, from: usize, to: usize) -> bool {
        let no_duplicates = !self.edges[from].contains(&to);
        if no_duplicates {
            self.edges[from].push(to);
        }

        no_duplicates
    }

    fn reorder(&mut self, mut indices: Box<[usize]>) {

        self.edges_mut().for_each(
            |edge| *edge = indices.iter().position(|i| i == edge).unwrap()
        );

        self.permute(&mut indices);
    }

    pub fn connect(&mut self, from_index: usize, to_index: usize) -> Option<Box<[usize]>> {

        if !self.try_connect_indexes(from_index, to_index) {
            return None;
        }

        topological_sort(&self.edges).map(|indices| {
            self.reorder(indices.clone());
            indices
        }).or_else(|| {
            self.edges[from_index].pop();
            None
        })
    }
}