pub fn find_remove<T: Eq>(vec: &mut Vec<T>, object: &T) {
    let pos = vec.iter().position(|e| e == object).unwrap();
    vec.remove(pos);
}

pub fn has_duplicates<T: Eq>(slice: &[T]) -> bool {

    for (i, e) in slice.iter().enumerate() {
        for (j, o) in slice.iter().enumerate() {
            if o == e && i != j {
                return true;
            }
        }
    }
    false
}

#[allow(clippy::len_without_is_empty)]
pub trait Permute {
    fn swap(&mut self, i: usize, n: usize);
    fn len(&self) -> usize;

    fn permute(&mut self, indices: &[usize]) {
        let len = self.len();
        assert_eq!(len, indices.len(), "slices must have the same length");
        assert!(!has_duplicates(indices), "indices must not have duplicates");
        assert!(indices.iter().all(|i| i < &len), "all indices must be valid");

        let mut indices = Box::<[usize]>::from(indices);

        for i in 0..indices.len() {

            let mut current = i;

            while i != indices[current] {

                let next = indices[current];
                self.swap(current, next);

                indices[current] = current;
                current = next;
            }

            indices[current] = current;
        }
    }
}

impl<T> Permute for [T] {
    fn swap(&mut self, i: usize, n: usize) {
        self.swap(i, n);
    }
    fn len(&self) -> usize {
        self.len()
    }
}