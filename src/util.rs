

use nih_plug_egui::egui::*;
use crate::gui::widgets::TooltipConfig;

pub fn find_remove<T: Eq>(vec: &mut Vec<T>, object: &T) {
    let pos = vec.iter().position(|e| e == object).unwrap();
    vec.remove(pos);
}

pub fn has_duplicates<T: Eq>(slice: &[T]) -> bool {

    slice
        .iter()
        .enumerate()
        .all(|(i, e)| slice.iter().position(|el| el == e).unwrap() == i)
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

pub fn param_tooltip(
    painter: &Painter,
    TooltipConfig { bg_col, text_col, border_stroke, font, position, offset, .. }: TooltipConfig,
    widget_rect: &Rect,
    text: String,
) {
    let pos = position.pos_in_rect(widget_rect) + offset * position.to_sign();
    let expand = Vec2::splat(font.size) * vec2(0.5, 0.15);
    let galley = painter.layout_no_wrap(text, font, text_col);

    let rect = invert_anchor(position).anchor_rect(Rect::from_min_size(pos, galley.size() + expand * 2.));
    painter.rect(rect, Rounding::same(8.), bg_col, border_stroke);
    painter.galley(rect.shrink2(expand).min, galley);
}

pub fn invert_align(align: Align) -> Align {
    match align {
        Align::Min => Align::Max,
        Align::Center => Align::Center,
        Align::Max => Align::Min,
    }
}

pub fn invert_anchor(anchor: Align2) -> Align2 {
    Align2([
        invert_align(anchor[0]),
        invert_align(anchor[1])
    ])
}

pub fn animate_draggable(painter: &Painter, response: &Response) -> f32 {
    painter.ctx().animate_bool(response.id, response.dragged() || response.hovered())
}

// TODO  quantized parameter boxes