use std::f32::consts::PI;
use std::ops::Range;
use std::hash::Hash;

use nih_plug::prelude::*;
use nih_plug_egui::egui::*;
use epaint::PathShape;
pub use plot::*;

/// Some redundant logic found in all draggable widgets
fn handle_drag(response: &Response, param: &impl Param, setter: &ParamSetter) {

    let begin = || setter.begin_set_parameter(param);
    let end = || setter.end_set_parameter(param);

    if response.drag_started() {
        begin()
    } else if response.drag_released() {
        end()
    }

    if response.double_clicked() {
        begin();
        setter.set_parameter(param, param.default_plain_value());
        end();
    }
}

fn animate_draggable(ui: &Ui, response: &Response) -> f32 {
    ui.ctx().animate_bool(response.id, response.dragged() || response.hovered())
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Orientation {
    Vertical,
    Horizontal
}

fn change_value(
    setter: &ParamSetter,
    param: &impl Param,
    response: &mut Response,
    sensitivity: f32,
    orientation: Orientation
) {

    let delta = response.drag_delta();
    if delta != Vec2::ZERO {

        response.mark_changed();
        let val = param.unmodulated_normalized_value();
        let drag = if orientation == Orientation::Vertical { delta.y } else { delta.x };
        setter.set_parameter_normalized(param, val + drag * sensitivity);
    }
}

pub struct Knob<'a, T> {
    col: Color32,
    r: f32,
    param: &'a T,
    setter: &'a ParamSetter<'a>,
    arc_start: f32,
}

impl<'a, T: Param> Knob<'a, T> {
    pub fn new(param: &'a T, setter: &'a ParamSetter) -> Self {
        Self { col: Color32::RED, r: 25., arc_start: 0., param, setter }
    }

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

impl<'a, T: Param> Widget for Knob<'a, T> {

    fn ui(self, ui: &mut Ui) -> Response {

        let Knob { col, r, param, setter, arc_start } = self;

        let s = r * 2.;
        let knob_size = Vec2::splat(s);
        let text_size = ui.text_style_height(&TextStyle::Body);
        let text_rect = vec2(0., text_size);
        let (rect, mut response) = ui.allocate_exact_size(knob_size + text_rect, Sense::click_and_drag());
        let knob_rect = Rect::from_min_size(rect.min, knob_size);

        if ui.is_rect_visible(rect) {

            handle_drag(&response, param, setter);
            change_value(setter, param, &mut response, -0.006, Orientation::Vertical);

            let knob_center = knob_rect.center();

            let painter = ui.painter();

            painter.text(
                knob_center + vec2(0., (s + text_size) / 2.),
                Align2::CENTER_CENTER,
                param.name().to_uppercase(),
                FontId { size: text_size, family: FontFamily::Proportional },
                ui.style().visuals.text_color()
            );

            let val = param.unmodulated_normalized_value();

            let width = r / 16. * (1.25 + 0.75 * animate_draggable(ui, &response));

            painter.circle_filled(knob_center, r * 0.75, Color32::DARK_GRAY);
            let arrow = Vec2::angled(3. * PI * (1. / 4. + val * 1. / 2.)) * r;

            let precision = r * 10.;
            let arc_radius = r - width / 2.;
            let mut start = (precision * arc_start) as usize;
            let mut end = (precision * val) as usize;

            if start > end {
                (start, end) = (end, start);
            }

            // TODO: I know there is a better way to do this with bezier curves.
            let colored_points = (start..end).map(
                |n| knob_center + Vec2::angled(PI * (0.75 + 1.5 * n as f32 / precision)) * arc_radius
            ).collect();

            painter.add(Shape::Path(PathShape::line(colored_points, Stroke::new(width, col))));

            painter.line_segment(
                [knob_center + arrow * 0.4, knob_center + arrow], 
                Stroke::new(width, Color32::WHITE)
            );
        }
        response
    }
}

pub struct HSlider<'a, T: Param> {
    param: &'a T,
    setter: &'a ParamSetter<'a>,
    padding: [f32 ; 2],
    invert: bool
}

impl<'a, T: Param> HSlider<'a, T> {
    pub fn for_param(param: &'a T, setter: &'a ParamSetter) -> Self {
        Self { param, setter, padding: [0. ; 2], invert: false }
    }

    pub fn with_padding(mut self, padding: [f32 ; 2]) -> Self {
        self.padding = padding;
        self
    }

    pub fn inverted(mut self) -> Self {
        self.invert = !self.invert;
        self
    }
}

impl<'a, T: Param> Widget for HSlider<'a, T> {

    fn ui(self, ui: &mut Ui) -> Response {

        let Self { param, setter, padding, invert } = self;

        let (mut rect, mut response) = ui.allocate_exact_size(
            vec2(ui.available_width(), 12.),
            Sense::click_and_drag()
        );

        rect.min.x += padding[0];
        rect.max.x -= padding[1];

        if ui.is_rect_visible(rect) {

            let width = rect.width();

            handle_drag(&mut response, param, setter);

            let direction = if invert { -1. } else { 1. };

            change_value(setter, param, &mut response, direction / width, Orientation::Horizontal);

            let height = 8. + 4. * animate_draggable(ui, &response);

            rect = Rect::from_center_size(rect.center(), vec2(width, height));
            let mut filled = rect;

            let pos = param.modulated_normalized_value() * width;

            let center_pos = if invert {
                rect.right() - pos
            } else {
                rect.left() + pos
            };

            rect.set_left(center_pos);
            filled.set_right(center_pos);

            let painter = ui.painter();
            let r = Rounding::same(3.);

            painter.rect_filled(filled, r, Color32::from_rgb(160, 160, 160));
            painter.rect_filled(rect, r, Color32::DARK_GRAY);

            let button_half_width = width / 9.;

            let button_rect = Rect::from_two_pos(
                pos2(filled.min.x.max(center_pos - button_half_width), filled.min.y),
                pos2(rect.max.x.min(center_pos + button_half_width), rect.max.y)
            );

            painter.rect_filled(button_rect, r, Color32::WHITE);
        }
        response
    }
}

/// Vertival slider
pub struct VSlider<'a, T: Param> {
    param: &'a T,
    setter: &'a ParamSetter<'a>,
    padding: [f32 ; 2],
    invert: bool
}

impl<'a, T: Param> VSlider<'a, T> {
    pub fn for_param(param: &'a T, setter: &'a ParamSetter) -> Self {
        Self { param, setter, padding: [0., 0.], invert: false }
    }

    pub fn with_padding(mut self, padding: [f32 ; 2]) -> Self {
        self.padding = padding;
        self
    }

    pub fn inverted(mut self) -> Self {
        self.invert = !self.invert;
        self
    }
}

impl<'a, T: Param> Widget for VSlider<'a, T> {

    fn ui(self, ui: &mut Ui) -> Response {

        let Self { param, setter, padding, invert } = self;

        let (mut rect, mut response) = ui.allocate_exact_size(
            vec2(12., ui.available_height()),
            Sense::click_and_drag()
        );
        rect.min.y += padding[0];
        rect.max.y -= padding[1];

        if ui.is_rect_visible(rect) {

            let height = rect.height();

            handle_drag(&mut response, param, setter);

            let direction = if invert { 1. } else { -1. };

            change_value(setter, param, &mut response, direction / height, Orientation::Vertical);

            let width = 8. + 4. * animate_draggable(ui, &response);

            rect = Rect::from_center_size(rect.center(), vec2(width, height));
            let mut filled = rect;

            let pos = param.modulated_normalized_value() * height;

            let center_pos = if invert {
                rect.top() + pos
            } else {
                rect.bottom() - pos
            };

            rect.set_bottom(center_pos);
            filled.set_top(center_pos);

            let painter = ui.painter();
            let r = Rounding::same(3.);

            painter.rect_filled(filled, r, Color32::from_rgb(160, 160, 160));
            painter.rect_filled(rect, r, Color32::DARK_GRAY);

            let button_half_height = height / 9.;

            let button_rect = Rect::from_two_pos(
                pos2(rect.min.x, rect.min.y.max(center_pos - button_half_height)),
                pos2(filled.max.x, filled.max.y.min(center_pos + button_half_height))
            );

            painter.rect_filled(button_rect, r, Color32::WHITE);
        }
        response
    }
}

/// Plain old plot. Show only the curve, without any other distractions.
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
}

// TODO  quantized parameter boxes