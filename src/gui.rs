pub mod widgets;
pub mod util;

use nih_plug_egui::egui::{Color32, Stroke, FontId, Align2, FontFamily, Painter, Rect, Vec2, vec2, Rounding, Align, Response, pos2, Widget, Ui, epaint, Sense, LayerId, Order, plot};

pub struct TooltipConfig {
    pub bg_col: Color32,
    pub text_col: Color32,
    pub border_stroke: Stroke,
    pub font: FontId,
    pub position: Align2,
    pub offset: f32,
    pub show_name: bool
}

impl Default for TooltipConfig {
    fn default() -> Self {
        TooltipConfig {
            bg_col: Color32::from_rgb(20, 20, 20),
            text_col: Color32::LIGHT_GRAY,
            border_stroke: Stroke::new(0.5, Color32::WHITE),
            font: FontId { size: 18., family: FontFamily::Proportional },
            position: Align2::CENTER_BOTTOM,
            offset: 5.,
            show_name: true,
        }
    }
}