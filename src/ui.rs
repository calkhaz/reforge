use winit::window::Window;

pub struct ui {
    ctx: egui::Context,
    state: egui_winit::State,
    window_input: egui::RawInput,
    platform_output: egui::PlatformOutput,
}

impl ui {
    pub fn new(window: &Window) -> ui {
        let ctx = egui::Context::default();
        ctx.set_visuals(egui::Visuals::dark());

        // Setting none for max_texture_side for now
        let state = egui_winit::State::new(ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, None);
        
        ui {
            ctx, state, window_input: egui::RawInput::default(), platform_output: egui::PlatformOutput::default()
        }
    }

    pub fn process_window_input(&mut self, window: &Window) {
        self.window_input = self.state.take_egui_input(&window);
    }

    pub fn run(&mut self) {
        let window_input = std::mem::replace(&mut self.window_input, egui::RawInput::default());

        let full_output: egui::FullOutput = self.ctx.run(window_input, |ctx| {
            egui::Window::new("Egui test")
                    .show(ctx, |ui| {
                ui.label("General test for egui working");
            });
        });

        let clipped_primitives = self.ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        self.platform_output = full_output.platform_output;

        //println!("{:?}", clipped_primitives);
    }

    pub fn handle_ui_window_event(&mut self, window: &Window) {
        let platform_output = std::mem::replace(&mut self.platform_output, egui::PlatformOutput::default());

        self.state.handle_platform_output(&window, platform_output);
    }
}

