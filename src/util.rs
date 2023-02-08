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

// TODO  quantized parameter boxes