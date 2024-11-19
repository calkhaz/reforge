use winit::window::Window;

pub struct Ui {
    ctx: egui::Context,
    state: egui_winit::State,
    window_input: egui::RawInput,
    platform_output: egui::PlatformOutput,
    pub hidden: bool
}

impl Ui {
    pub fn new(window: &Window) -> Ui {
        let ctx = egui::Context::default();
        ctx.set_visuals(egui::Visuals::dark());

        // Setting none for max_texture_side for now
        let state = egui_winit::State::new(ctx.clone(), egui::ViewportId::ROOT, &window, Some(window.scale_factor() as f32), None, None);
        
        Ui {
            ctx, state, window_input: egui::RawInput::default(), platform_output: egui::PlatformOutput::default(), hidden: false
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
            let win = egui::Window::new("Egui");
                win.show(ctx, |ui| {
                ui.label("Is egui working?");
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

