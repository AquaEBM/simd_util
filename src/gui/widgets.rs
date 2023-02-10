use super::*;
use epaint::{Shape, PathShape};
use plot::Plot;
use crate::parameter::Parameter;
use util::{param_tooltip, animate_draggable};
use std::{f32::consts::PI, ops::Range, hash::Hash};

pub trait DraggableWidget {

    fn size(&self) -> [Option<f32> ; 2];
    fn drag_sensitivity(&self) -> Vec2;
    fn orientation(&self) -> bool;

    fn draw(self, ui: &mut Ui, response: &mut Response, norm_val: f32, param: &impl Parameter);

    fn tooltip_options(&self) -> TooltipConfig { Default::default() }
}

pub struct ParamWidget<T, U> {
    widget: T,
    param: U,
}

impl<T, U> ParamWidget<T, U> {
    pub fn new(widget: T, param: U) -> Self { Self { widget, param } }
}

impl <T: Default, U> ParamWidget<T, U> {
    pub fn default(param: U) -> Self { Self { widget: T::default(), param } }
}

impl<T: DraggableWidget, U: Parameter> Widget for ParamWidget<T, U> {
    fn ui(self, ui: &mut Ui) -> Response {

        let Self { widget, param } = self;

        let [w, h] = widget.size();

        let size = vec2(
            w.unwrap_or_else(|| ui.available_width()),
            h.unwrap_or_else(|| ui.available_height())
        );

        let mut response = ui.allocate_response(size, Sense::click_and_drag());
        ui.set_clip_rect(Rect::EVERYTHING);

        let rect = response.rect;

        if response.drag_started() {
            param.begin_automation();
        } else if response.drag_released() {
            param.end_automation();
        }

        let ctx = ui.ctx();

        let response_id = response.id;
        let param_id = param.id().unwrap_or(response_id);

        if response.double_clicked() {
            param.begin_automation();

            let default = param.default_normalized_value();

            *ctx.data().get_temp_mut_or(param_id, default) = default;

            param.set_normalized_value(default);
            
            param.end_automation();
        }

        let delta = response.drag_delta();
        
        if delta != Vec2::ZERO {

            let scaled_delta = widget.drag_sensitivity() * delta * vec2(1., -1.) / rect.size();
            let diff = scaled_delta[widget.orientation() as usize];

            let mut data = ctx.data();

            let cached_norm_val = data.get_temp_mut_or_insert_with(
                param_id,
                || param.get_normalized_value()
            );

            let new_val = (*cached_norm_val + diff).clamp(0., 1.);

            *cached_norm_val = new_val;

            param.set_normalized_value(new_val);

            response.mark_changed();
        }

        let smoothed_norm_val = ctx.animate_value_with_time(
            response_id,
            param.get_normalized_value(),
            0.05
        );
        let name = param.name();

        println!("{name}: {}", smoothed_norm_val);

        let tooltip_options = widget.tooltip_options();

        if ui.is_rect_visible(rect) {

            widget.draw(ui, &mut response, smoothed_norm_val, &param);
        }

        if response.dragged() {

            let value_string = param.norm_val_to_string(smoothed_norm_val);

            let text = if true {
                format!("{}: {}", param.name(), value_string)
            } else {
                value_string
            };

            let painter = ui.painter().clone();
            let layer_id = painter.layer_id().id;

            param_tooltip(
                &painter.with_layer_id(LayerId ::new(Order::Tooltip, layer_id)),
                tooltip_options,
                &rect,
                text
            );
        }
        
        response
    }
}

pub struct Knob {
    col: Color32,
    r: f32,
    arc_start: f32,
}

impl Default for Knob {
    fn default() -> Self {
        Self { col: Color32::RED, r: 20., arc_start: 0. }
    }
}

impl Knob {
    
    pub fn new() -> Self { Self::default() }
    
    pub fn color(mut self, col: impl Into<Color32>) -> Self {
        self.col = col.into();
        self
    }
    
    pub fn radius(mut self, r: f32) -> Self {
        self.r = r;
        self
    }
    
    pub fn arc_start(mut self, value: f32) -> Self {
        self.arc_start = value;
        self
    }
}

impl DraggableWidget for Knob {
    
    fn size(&self) -> [Option<f32> ; 2] { [Some(self.r * 2.), Some(self.r * 2.7)] }

    fn drag_sensitivity(&self) -> Vec2 { vec2(0., 0.5) }

    fn orientation(&self) -> bool { true }
    
    fn draw(self, ui: &mut Ui, response: &mut Response, norm_val: f32, param: &impl Parameter) {

        let painter = ui.painter();

        let Self { arc_start, col, r } = self;

        let knob_center = response.rect.center_top() + vec2(0., r);

        let text_size = r * 0.8;

        painter.text(
            knob_center + vec2(0., r + text_size / 2.),
            Align2::CENTER_CENTER,
            param.name().to_uppercase(),
            FontId { size: text_size, family: FontFamily::Proportional },
            Color32::LIGHT_GRAY
        );
        
        let width = r / 15. * (1.25 + 0.75 * animate_draggable(painter, &response));
        
        painter.circle_filled(knob_center, r * 0.75, Color32::DARK_GRAY);
        let arrow = Vec2::angled(1.5 * PI * (0.5 + norm_val)) * r;
        
        let precision = r * 10.;
        let arc_radius = r - width / 2.;
        
        let mut start = (precision * arc_start) as usize;
        let mut end = (precision * norm_val) as usize;
        if start > end { (start, end) = (end, start); }
        
        let colored_points = (start..end).map(
            |n| knob_center + Vec2::angled(PI * (0.75 + 1.5 * n as f32 / precision)) * arc_radius
        ).collect();
        
        painter.add(Shape::Path(PathShape::line(colored_points, Stroke::new(width, col))));
        
        painter.line_segment(
            [knob_center + arrow * 0.4, knob_center + arrow], 
            Stroke::new(width, Color32::WHITE)
        );
    }
}

pub struct HSlider {
    padding: [f32 ; 2],
    invert: bool
}

impl Default for HSlider {
    fn default() -> Self {
        Self { padding: [0., 0.], invert: false }
    }
}

impl HSlider {
    
    pub fn new() -> Self { Self::default() }
    
    pub fn with_padding(mut self, padding: [f32 ; 2]) -> Self {
        self.padding = padding;
        self
    }
    
    pub fn inverted(mut self) -> Self {
        self.invert = !self.invert;
        self
    }
}

impl DraggableWidget for HSlider {
    
    fn size(&self) -> [Option<f32> ; 2] { [None, Some(12.)] }
    
    fn drag_sensitivity(&self) -> Vec2 { vec2(if self.invert { -1. } else { 1. }, 0.) }
    
    fn orientation(&self) -> bool { false }
    
    fn draw(self, ui: &mut Ui, response: &mut Response, norm_val: f32, _param: &impl Parameter) {
        
        let Self { padding: [pad_l, pad_r], invert } = self;
        
        let mut rect = response.rect;
        
        rect.min.x += pad_l;
        rect.max.x -= pad_r;

        let painter = ui.painter();
                
        let thickness = 0.15 * (1. - animate_draggable(painter, &response));
        
        let width = rect.width();
        let height = rect.height();
        
        let mut filled_part = rect.shrink2(vec2(0., height * thickness));
        let mut empty_part = filled_part;
        
        let pos = norm_val * width;
        
        let center_pos = if invert {
            filled_part.right() - pos
        } else {
            filled_part.left() + pos
        };
        
        empty_part.set_left(center_pos);
        filled_part.set_right(center_pos);
        
        let r = Rounding::same(3.);
        
        let button_half_width = width / 8.;
        
        let button_rect = Rect::from_two_pos(
            pos2(empty_part.min.y.max(center_pos - button_half_width), empty_part.min.y),
            pos2(filled_part.max.y.min(center_pos + button_half_width), filled_part.max.x)
        );
        
        painter.rect_filled(filled_part, r, Color32::GRAY);
        painter.rect_filled(empty_part, r, Color32::DARK_GRAY);
        painter.rect_filled(button_rect, r, Color32::WHITE);
    }
}

/// Vertical slider
pub struct VSlider {
    padding: [f32 ; 2],
    invert: bool
}

impl Default for VSlider {
    fn default() -> Self {
        Self { padding: [0., 0.], invert: false }
    }
}

impl VSlider {
    
    pub fn new() -> Self { Self::default() }
    
    pub fn with_padding(mut self, padding: [f32 ; 2]) -> Self {
        self.padding = padding;
        self
    }
    
    pub fn inverted(mut self) -> Self {
        self.invert = !self.invert;
        self
    }
}

impl DraggableWidget for VSlider {
    
    fn size(&self) -> [Option<f32> ; 2] { [Some(12.), None] }
    
    fn drag_sensitivity(&self) -> Vec2 { vec2(0., if self.invert { -1. } else { 1. }) }
    
    fn orientation(&self) -> bool { true }
    
    fn draw(self, ui: &mut Ui, response: &mut Response, norm_val: f32, _param: &impl Parameter) {
        
        let Self { padding: [pad_u, pad_d], invert } = self;
        
        let mut rect = response.rect;
        
        rect.min.y += pad_u;
        rect.max.y -= pad_d;

        let painter = ui.painter();
        
        let thickness = 0.15 * (1. - animate_draggable(painter, &response));
        
        let height = rect.height();
        let width = rect.width();
        
        let mut filled_part = rect.shrink2(vec2(width * thickness, 0.));
        let mut empty_part = filled_part;
        
        let pos = norm_val * height;
        
        let center_pos = if invert {
            filled_part.top() + pos
        } else {
            filled_part.bottom() - pos
        };
        
        empty_part.set_bottom(center_pos);
        filled_part.set_top(center_pos);
        
        let r = Rounding::same(3.);
        
        let button_half_height = height / 8.;
        
        let button_rect = Rect::from_two_pos(
            pos2(empty_part.min.x, empty_part.min.y.max(center_pos - button_half_height)),
            pos2(filled_part.max.x, filled_part.max.y.min(center_pos + button_half_height))
        );
        
        painter.rect_filled(filled_part, r, Color32::GRAY);
        painter.rect_filled(empty_part, r, Color32::DARK_GRAY);
        painter.rect_filled(button_rect, r, Color32::WHITE);
    }
}

/// Plain old plot. Show only the curve, no other distractions.
// TODO: create our own plot, because this plot still reacts to some interactions (double-clicking)
// and the way we use it doesn't require storing an ID.
pub fn plain_plot(id: impl Hash, x_range: Range<f64>, y_range: Range<f64>) -> Plot {
    Plot::new(id)
    .allow_boxed_zoom(false)
    .allow_drag(false)
    .allow_scroll(false)
    .allow_zoom(false)
    .show_x(false)
    .show_y(false)
    .show_axes([false, false])
    .include_x(x_range.start)
    .include_x(x_range.end)
    .include_y(y_range.start)
    .include_y(y_range.end)
    .view_aspect(2.)
}