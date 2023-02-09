use std::{ops::{Deref, DerefMut}, borrow::Borrow};
use crate::util::{find_remove, permute};


fn position(edges: &[Edge], index: &usize) -> Option<usize> {
    edges.iter().position(|(Edge::Normal(i) | Edge::Feedback(i))| i == index)
}

    incoming_edges.iter().all(Vec::is_empty).then_some(new_order)
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
            &Self::Normal(i) => Some(i),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
pub struct Graph<I, D> {
    nodes: Vec<D>,
    ids: Vec<I>,
    edges: Vec<Vec<Edge>>,
}

impl<I, D> Graph<I, D> {

    pub fn top_level_insert(&mut self, id: I, data: D) {
        self.nodes.push(data);
        self.ids.push(id);
        self.edges.push(vec![]);
    }

    fn connect_indexes(&mut self, from: usize, to: usize) -> bool {
        let already_connected = self.edges[from].iter().any(|(Edge::Feedback(i) | Edge::Normal(i))| i == &to);
        if already_connected {
            self.edges[from].push(Edge::Normal(to));
        }
        already_connected
    }

    fn non_feedback_edges(&self) -> Vec<Vec<usize>> {
        self.edges.iter().map(|edges| edges.iter()
            .filter_map(Edge::index_if_normal)
            .collect()
        ).collect()
    }

    pub fn find_node<Q>(&mut self, node_id: &Q) -> usize
    where
        I: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.ids.iter().position(|id| id.borrow() == node_id).unwrap()
    }

    pub fn connect<Q>(&mut self, from_id: &Q, to_id: &Q)
    where
        I: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let from_index = self.find_node(from_id);
        let to_index = self.find_node(to_id);

        self.connect_indexes(from_index, to_index);

        topological_sort(self.non_feedback_edges()).or_else(|| {
            self.edges[from_index].last_mut().unwrap().set_as_feedback();
            None
        });
    }
}