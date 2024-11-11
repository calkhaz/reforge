mod config;
mod ffmpeg;
mod py;
mod reforge;
mod render;
mod ui;
mod utils;
mod vulkan;

use ash::vk;
use anyhow::{anyhow, Context, Result};
use clap::Parser;
use tracing::{debug, warn};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};
use reforge::Reforge;


#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
enum ShaderFormat {
    Rgba8,
    Rgba32f
}

impl ShaderFormat {
    fn to_vk_format(self) -> vk::Format {
        match self {
            ShaderFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
            ShaderFormat::Rgba32f => vk::Format::R32G32B32A32_SFLOAT
        }
    }
}

#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error
}

impl LogLevel {
    pub fn to_trace(&self) -> tracing::Level {
        match self {
            LogLevel::Trace => tracing::Level::TRACE,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Info  => tracing::Level::INFO,
            LogLevel::Warn  => tracing::Level::WARN,
            LogLevel::Error => tracing::Level::ERROR,
        }
    }
}

#[derive(clap::Parser, Clone, Default, Debug)]
pub struct Args {
    #[arg(short='i', long="input-file", help = "File to read from")]
    input_file: Option<String>,

    #[arg(short='o', long="output-file", help = "File to write to. A window preview is used otherwise")]
    output_file: Option<String>,

    #[arg(short='p', long="py", help = "Python config file [Absolute or relative to py-config-path (.py optional)]")]
    python_config: Option<String>,

    #[arg(short='P', long="py-config-path", help = "Python config path [Defaults to $REFORGE_PY_CONFIG_PATH]")]
    python_path: Option<String>,

    #[arg(short='S', long="shader-path", help = "Path to find shaders [Defaults to $REFORGE_SHADER_PATH]")]
    shader_path: Option<String>,

    #[arg(short='s', long="shader-file", help = "Direct path to a shader file")]
    shader_file: Option<String>,

    #[arg(long)]
    width: Option<u32>,

    #[arg(long)]
    height: Option<u32>,

    #[arg(long, default_value = "rgba32f", help = "Shader image format")]
    shader_format: Option<ShaderFormat>,

    #[arg(long, value_name="config", help = "Path to the pipeline configuration file")]
    config: Option<String>,

    #[arg(long, default_value= "2", help = "Number of frame-in-flight to be used when displaying to the swapchain")]
    num_frames: Option<usize>,

    #[arg(short='l', long="log-level", help = "Tracing log level")]
    log_level: Option<LogLevel>,
}

#[derive(Default)]
struct WindowHandler {
    args: Args,
    reforge: Option<Reforge>,
    close_requested: bool,
    window: Option<Window>,
}

impl WindowHandler {
    fn new(args: Args) -> WindowHandler {
        WindowHandler {
            args,
            ..Default::default()
        }
    }
}

// Validation checks and adjust args
fn update_args(mut args: Args) -> Result<Args> {
    if args.shader_file.is_none() && args.shader_path.is_none() {
        let env_path = std::env::var("REFORGE_SHADER_PATH").context("Missing required --shader-path, --shader-file or env var REFORGE_SHADER_PATH")?;
        debug!("Setting shader_path via env to: {}", env_path);
        args.shader_path = Some(env_path);
    }

    if let Some(python_config) = args.python_config.as_ref() {
        args.python_config = Some(utils::find_python_config(&python_config, args.python_path.clone())?);
    }

    if args.shader_file.is_none() && args.python_config.is_none() {
        return Err(anyhow!("Expected either --shader-file or --python-config"))
    }

    Ok(args)
}

impl ApplicationHandler for WindowHandler {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let (width, height) = utils::get_dim(800, 600, self.args.width, self.args.height);

        let inner_size = winit::dpi::Size::new(winit::dpi::PhysicalSize::new(width, height));

        let attr = winit::window::WindowAttributes::default()
            .with_inner_size(inner_size)
            .with_title("Reforge");

        self.window = Some(event_loop.create_window(attr).unwrap());

        self.args.width = Some(width);
        self.args.height = Some(height);

        if self.reforge.is_none() {
            self.reforge = Some(Reforge::new(self.args.clone(), self.window.as_ref()).unwrap());
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => self.close_requested = true,
            WindowEvent::KeyboardInput {
                event: KeyEvent { logical_key: key, state: ElementState::Pressed, .. },
                ..
            } => match key.as_ref() {
                Key::Named(NamedKey::Escape) => {
                    self.close_requested = true;
                },
                _ => (),
            },
            WindowEvent::Resized(size) => {
                self.reforge.as_mut().unwrap().resize_swapchain(size.width, size.height).unwrap();
            },
            WindowEvent::RedrawRequested => {
                let window = self.window.as_ref().unwrap();
                window.pre_present_notify();

                if let Some(rf) = &mut self.reforge {
                    rf.render.ui.as_mut().unwrap().process_window_input(window);

                    if let Err(err) = rf.run() {
                        warn!("Rendering err: {err}");
                    }

                    rf.render.ui.as_mut().unwrap().handle_ui_window_event(window);
                }
            },
            _ => (),
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        match self.close_requested {
            true  => event_loop.exit(),
            false => self.window.as_ref().unwrap().request_redraw()
        }
    }
}

fn run_reforge(args: Args) -> Result<()> {
    let log_level = args.log_level.unwrap_or(LogLevel::Warn).to_trace();

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(log_level).finish();

    let _ = tracing::subscriber::set_global_default(subscriber).context("setting tracing default failed");

    let span = tracing::trace_span!("reforge");
    let _guard = span.enter();

    let args = update_args(args)?;

    if args.output_file.is_some() {
        let mut reforge = Reforge::new(args, None)?;

        loop {
            match reforge.run() {
                Ok(finished) => if finished { break; },
                Err(err) => return Err(err)
            }
        }

        Ok(())
    }
    else {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut app = WindowHandler::new(args);
        Ok(event_loop.run_app(&mut app)?)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    run_reforge(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_args(args_str: &str) -> Args {
        let prefixed_string = format!("reforge {args_str}");
        let split_args: Vec<&str> = prefixed_string.split_whitespace().collect();

        Args::parse_from(split_args)
    }

    fn compare_images(candidate: &str, reference: &str) -> Result<f64> {
        // Seems using ffmpeg cmd to externally write the file
        // causes us a race condition where the file may not be fully written
        // No amount of flushing/manual closing seems to help, so we wait a short moment before
        // opening the files we just finished writing
        std::thread::sleep(std::time::Duration::from_millis(100));
        let c_bytes = std::fs::read(candidate).context(format!("Error reading: {candidate}"))?;
        let r_bytes = std::fs::read(reference).context(format!("Error reading: {reference}"))?;

        if c_bytes.len() == r_bytes.len() {
            if c_bytes == r_bytes {
                return Ok(1.0)
            }
        }

        let c_data = image::ImageReader::new(std::io::Cursor::new(c_bytes.clone())).with_guessed_format()?.decode()?.into_rgba8();
        let r_data = image::ImageReader::new(std::io::Cursor::new(r_bytes.clone())).with_guessed_format()?.decode()?.into_rgba8();

        let compare = image_compare::rgba_hybrid_compare(&c_data, &r_data)?;

        Ok(compare.score)
    }

    fn make_io(input: &str) -> (String, String, String) {
        let func_name = std::thread::current().name().unwrap().to_string();
        (input.to_string(),
        format!("tests/candidate-images/{func_name}.png"),
        format!("tests/reference-images/{func_name}.png"))
    }

    fn test_single_io() -> Result<f64> {
        let (input, candidate, reference) = make_io("tests/images/waterfall.jpg");
        let args = make_args(&format!("-i {input} --shader-file tests/shaders/passthrough.comp -o {candidate}"));
        let _ = run_reforge(args)?;

        compare_images(&candidate, &reference)
    }

    fn test_chaining_io() -> Result<f64> {
        let (input, candidate, reference) = make_io("tests/images/waterfall.jpg");
        let args = make_args(&format!("-i {input} --shader-path tests/shaders --py-config-path tests/py -p chaining_io -o {candidate}"));
        let _ = run_reforge(args)?;

        compare_images(&candidate, &reference)
    }

    fn test_gen() -> Result<f64> {
        let (_, candidate, reference) = make_io("");
        let args = make_args(&format!("--shader-file tests/shaders/noise.comp -o {candidate}"));
        let _ = run_reforge(args)?;

        compare_images(&candidate, &reference)
    }

    fn test_point_op() -> Result<f64> {
        let (input, candidate, reference) = make_io("tests/images/waterfall.jpg");
        let args = make_args(&format!("-i {input} --py-config-path tests/py -p point_op --shader-path tests/shaders/ -o {candidate}"));
        let _ = run_reforge(args)?;

        compare_images(&candidate, &reference)
    }

    fn test_buffer_chain() -> Result<f64> {
        let (input, candidate, reference) = make_io("tests/images/waterfall.jpg");
        let args = make_args(&format!("-i {input} --py-config-path tests/py -p buffer_chain --shader-path tests/shaders/ -o {candidate}"));
        let _ = run_reforge(args)?;

        compare_images(&candidate, &reference)
    }

    fn test_compare_res(compare: Result<f64>) {
        if let Err(err) = &compare {
            eprintln!("{:?}", err);
        }
        assert!(compare.is_ok());
        assert!(compare.unwrap() > 0.95);
    }

    #[test] fn single_io()   { test_compare_res(test_single_io()) }
    #[test] fn chaining_io() { test_compare_res(test_chaining_io()) }
    #[test] fn gen() { test_compare_res(test_gen()) }
    #[test] fn point_op() { test_compare_res(test_point_op()) }
    #[test] fn buffer_chain() { test_compare_res(test_buffer_chain()) }
}
