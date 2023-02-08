use nih_plug_egui::egui::*;
use crate::widgets::TooltipConfig;

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