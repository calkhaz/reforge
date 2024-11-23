use winit::window::Window;
use crate::render::ParamData;

use std::collections::HashMap;
use tracing::warn;

#[derive(Clone, Debug)]
pub struct UiParam {
    pub min: ParamData,
    pub max: ParamData,
    pub val: ParamData
}

pub struct Ui {
    ctx: egui::Context,
    state: egui_winit::State,
    window_input: egui::RawInput,
    platform_output: egui::PlatformOutput,
    pub hidden: bool,
    pub params: HashMap<String, UiParam>
}

impl Ui {
    pub fn new(window: &Window) -> Ui {
        let ctx = egui::Context::default();
        ctx.set_visuals(egui::Visuals::dark());


        // Scale the default ui size down a bit
        ctx.set_pixels_per_point(0.8);

        // Setting none for max_texture_side for now
        let state = egui_winit::State::new(ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, None);
        
        Ui {
            ctx, state, window_input: egui::RawInput::default(), platform_output: egui::PlatformOutput::default(), hidden: false, params: HashMap::new()
        }
    }

    pub fn on_window_event(&mut self, window: &Window, event: &winit::event::WindowEvent) {
        let _ = self.state.on_window_event(window, event);
    }

    pub fn process_window_input(&mut self, window: &Window) {
        self.window_input = self.state.take_egui_input(&window);
    }

    pub fn run(&mut self) -> (Vec<egui::ClippedPrimitive>, egui::epaint::textures::TexturesDelta, f32) {
        let window_input = std::mem::replace(&mut self.window_input, egui::RawInput::default());

        let full_output: egui::FullOutput = self.ctx.run(window_input, |ctx| {
            let purple = egui::Color32::from_rgb(150, 123, 182);
            let dark = egui::Color32::from_rgba_unmultiplied(100, 100, 100, 60);
            let dark_highlight = egui::Color32::from_rgba_unmultiplied(130, 130, 130, 60);

            let shadow = egui::Shadow {
                offset: egui::Vec2{x: 1.0, y: 2.0},
                blur: 3.0,
                spread: 2.0,
                color: egui::Color32::from_rgba_unmultiplied(0, 0, 0, 120)
            };

            egui::Window::new("")
                .default_pos(egui::Pos2{x: 0.0, y: 0.0})
                .fade_in(true)
                .fade_out(true)
                .collapsible(true)
                .frame(egui::Frame::default()
                    .inner_margin(egui::Margin::same(10.0))
                    .fill(egui::Color32::from_rgba_unmultiplied(40, 40, 40, 180))
                    .shadow(egui::Shadow::default())
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(64, 64, 64, 120)))
                    .rounding(egui::Rounding::same(3.0))
                    .shadow(shadow)
                )
                .title_bar(false)
                .resizable(false)
                .movable(true)
                .show(ctx, |ui| {
                    ui.style_mut().visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(2.0, purple);

                    // Modify the style of the slider background and thumb
                    let style = ui.style_mut();
                    style.visuals.widgets.inactive.bg_fill = purple; // Set slider background color
                    style.visuals.widgets.hovered.bg_fill = purple;  // Set slider background color when hovered
                    style.visuals.widgets.active.bg_fill = purple;   // Set slider background color when active

                    // Customize slider value box
                    style.visuals.widgets.active.weak_bg_fill = dark;
                    style.visuals.widgets.inactive.weak_bg_fill = dark;
                    style.visuals.widgets.hovered.weak_bg_fill = dark;
                    style.visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, purple);
                    style.visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, purple);
                    style.visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, dark_highlight);
                    style.visuals.override_text_color = Some(egui::Color32::WHITE);

                    //style.visuals.button_frame = false;

                    for (param_name, p) in &mut self.params {
                        use ParamData::*;

                        // TODO: We can probably add a custom shadow or outline to text this way,
                        //       but we'll need to upload another font (creating new fontid) with different sizes
                        //       and replace  the TextStyles
                        // let resp: egui::Response = ui.label(" ");
                        // let x = resp.rect.min.x;
                        // let y = resp.rect.min.y;
                        // ui.painter().text(egui::Pos2::new(x, y), egui::Align2::LEFT_TOP, param_name, egui::TextStyle::Heading.resolve(&ui.style()), egui::Color32::from_rgba_premultiplied(255, 255, 255, 255));
                        // ui.painter().text(egui::Pos2::new(x, y), egui::Align2::LEFT_TOP, param_name, egui::TextStyle::Body.resolve(&ui.style()), egui::Color32::from_rgba_premultiplied(64, 64, 64, 255));

                        //ui.label("yep");
                        ui.label(egui::RichText::new(param_name).strong());

                        match (&mut p.val, p.min.clone(), p.max.clone()) {
                            (Float(v), Float(min), Float(max)) => {
                                ui.add(egui::Slider::new(v, min..=max).text_color(egui::Color32::from_rgb(255, 255, 255)));
                            },
                            (Integer(v), Integer(min), Integer(max)) => {
                                ui.add(egui::Slider::new(v, min..=max));
                            },
                            _ => {
                                warn!("Unsupported ui paramdata for {param_name}")
                            }
                        }
                    }

            });
        });

        let clipped_primitives = self.ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        self.platform_output = full_output.platform_output;

        (clipped_primitives, full_output.textures_delta, full_output.pixels_per_point)
    }

    pub fn handle_ui_window_event(&mut self, window: &Window) {
        let platform_output = std::mem::replace(&mut self.platform_output, egui::PlatformOutput::default());

        self.state.handle_platform_output(&window, platform_output);
    }
}

