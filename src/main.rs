extern crate ash;
extern crate clap;
extern crate gpu_allocator;
extern crate shaderc;
extern crate pyo3;
extern crate tracing;
extern crate tracing_subscriber;

mod config;
mod ffmpeg;
mod py;
mod render;
mod utils;
mod vulkan;

use ash::vk;
use anyhow::{anyhow, Context, Result};
use clap::Parser;
use ffmpeg::{Decoder, Encoder};
use render::Render;
use render::RenderInfo;
use tracing::{debug, info, warn};
use utils::TERM_CLEAR;

use crate::render::ParamData;

use std::collections::HashMap;

use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
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

#[derive(Copy, Clone, clap::ValueEnum)]
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

#[derive(clap::Parser, Default)]
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

struct Reforge {
    args: Args,
    width: u32,
    height: u32,
    decoder: Option<Decoder>,
    encoder: Option<Encoder>,
    render: Render,
    graph: String,
    params: HashMap<String, HashMap<String, ParamData>>,
    py_config_timestamp: u64,
    event_loop: Option<EventLoop<()>>,
    time_since_start: std::time::Instant,
}

impl Reforge {
    fn has_swapchain(&self) -> bool {
        self.event_loop.is_some()
    }

    fn write_params(&mut self) {
        self.params.iter().for_each(|(node_name, params)| {
            params.iter().for_each(|(param_name, value)| {
                let param_map = self.render.pipeline_buffer_data.entry(node_name.clone()).or_default();
                param_map.insert(param_name.clone(), value.clone());

                self.render.outdate_frames();
            })
        });
    }

    fn py_needs_reload(&mut self) -> bool {
        if let Some(python_config) = self.args.python_config.as_ref() {
            let current_py_config_timestamp = utils::get_modified_time(&python_config);

            if current_py_config_timestamp == 0 {
                warn!("Unable to access python config: {}", python_config);
            }

            if current_py_config_timestamp > self.py_config_timestamp {
                self.py_config_timestamp = current_py_config_timestamp;
                return true;
            }
        }

        false
    }

    fn reload_py_config(&mut self) -> Result<()> {
        if let Some(python_config) = self.args.python_config.as_ref() {
            let shader_path = self.args.shader_path.as_ref().unwrap().clone();
            match py::py_config(&python_config) {
                Ok((graph, params)) => {
                    self.params = params;

                    if graph != self.graph {
                        let config = config::parse(graph.clone(), &shader_path).context("Failed to create config")?;
                        self.render.update_config(config);
                        self.graph = graph;

                        debug!("Reload graph: {}", self.graph);
                    }

                    debug!("Reload Params: {:?}", self.params);

                    self.write_params();
                    Ok(())
                },
                Err(err) => Err(anyhow!("Python err in {}: {}", python_config, err))
            }
        }
        else {
            Ok(())
        }
    }

    fn new(args: Args) -> Result<Reforge> {
    
        let (mut width, mut height) = utils::get_dim(800, 600, args.width, args.height);
    
        let decoder = if let Some(input_file) = &args.input_file {
            let decoder = Decoder::new(input_file, args.width, args.height).context("Failed to create decoder")?;
            info!("Decoded size: {} X {}", decoder.width, decoder.height);
            width = decoder.width;
            height = decoder.height;
            Some(decoder)
        } else { None };
    
    
        let encoder = if let Some(output_file) = &args.output_file {
            Some(Encoder::new(&output_file, width, height).context("Failed to create encoder")?)
        } else { None };
    
        let (graph, params, py_config_timestamp) = if let Some(python_config) = args.python_config.as_ref() {
            let (graph, params) = py::py_config(&python_config)?;
            (graph, params, utils::get_modified_time(&python_config))
        }
        else {
            // Verified by this point in an earlier check
            let shader_file = args.shader_file.as_ref().unwrap();

            let graph = match decoder {
                Some(_) => format!("input -> {} -> output", shader_file),
                None    => format!(         "{} -> output", shader_file)
            };

            (graph, HashMap::new(), 0)
        };
    
        let shader_path = args.shader_path.clone().unwrap_or("".to_string());
        let config = config::parse(graph.clone(), &shader_path).context("Failed to create config")?;
        debug!("Graph: {}", graph);
        debug!("Config: {:?}", config);

        let render_info = RenderInfo {
            config,
            width,
            height,
            num_frames: args.num_frames.unwrap(),
            format: args.shader_format.unwrap().to_vk_format(),
            swapchain: encoder.is_none(),
            has_input_image: decoder.is_some(),
        };

        let use_swapchain = encoder.is_none();
        let event_loop = if use_swapchain { Some(EventLoop::new()) } else { None };
        let render = Render::new(render_info, &event_loop);
        let time_since_start: std::time::Instant = std::time::Instant::now();
    
        Ok(Reforge { args, width, height, decoder, encoder, render, graph, params, py_config_timestamp, event_loop, time_since_start })
    }

    pub fn execute(&mut self, input_bytes: Option<&[u8]>, output_bytes: Option<&mut [u8]>) -> bool {
        let mut first_run = vec![true; self.args.num_frames.unwrap()];

        //let mut avg_ms = 0.0;
        let mapped_input_image_data: *mut u8 = self.render.staging_buffer_ptr();

        // TODO: We really don't want to always write into the staging buffer
        //       if the contents haven't changed - 'first_run[]' needs to work together with run() to
        //       optimize this away
        // Write bytes into staging image
        if let Some(input_bytes) = input_bytes {
            unsafe { std::ptr::copy_nonoverlapping(input_bytes.as_ptr(), mapped_input_image_data, input_bytes.len()); }
        }

        let mut requested_exit = false;
        let mut first_resize = true;

        // Handle window events
        if let Some(event_loop) = &mut self.event_loop {
            event_loop.run_return(|event, _, control_flow| {
                *control_flow = ControlFlow::Poll;
                match event {
                    Event::WindowEvent {
                        event: WindowEvent::Resized(_size),
                        ..
                    } => {
                        // This event gets triggered on initial window creation
                        // and we don't want to recreate the swapchain at that point
                        if !first_resize {
                            self.render.swapchain_rebuilt_required = true;
                        }

                        first_resize = false;
                    },
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                            input: KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                            ..
                        },
                    .. } => {
                        *control_flow = ControlFlow::Exit;
                        requested_exit = true;
                    }
                    Event::MainEventsCleared => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                }
            });
        }


        // Wait for the previous iteration of this frame before
        // changing or executing on its resources
        self.render.wait_for_frame_fence();

        if self.render.trigger_reloads() {
            // Clear current line of timers
            eprint!("{TERM_CLEAR}");
            first_run.iter_mut().for_each(|b| *b = true);
        }

        self.render.update_ubos(self.time_since_start.elapsed().as_secs_f32());

        // Pull in the next image from the swapchain
        if self.has_swapchain() {
            self.render.acquire_swapchain();
        }

        self.render.begin_record();

        // On the first run, we:
        // 1. Transitioned images as needed
        // 2. Load the staging input buffer into an image and convert it to linear
        if first_run[self.render.frame_index] {
            if input_bytes.as_ref().is_some() {
                self.render.record_initial_image_load();
            }
            self.render.record_pipeline_image_transitions();
            //first_run[render.frame_index] = false;
        }

        self.render.record();

        if output_bytes.is_some() {
            self.render.write_output_to_buffer();
        }

        self.render.end_record();

        // Send the work to the gpu
        self.render.submit();

        if self.encoder.is_some() {
            self.render.wait_for_frame_fence();
    
            if let Some(output_bytes) = output_bytes {
                unsafe { std::ptr::copy_nonoverlapping(mapped_input_image_data, output_bytes.as_mut_ptr(), output_bytes.len()); }
            }
        }

        requested_exit
    }

    pub fn run(&mut self) -> Result<()> {
        let frame_size = self.width as usize * self.height as usize * 4;
        let mut window_exit_requested = false;

        let mut rf_output: Vec<u8> = Vec::with_capacity(frame_size);
        unsafe { rf_output.set_len(frame_size) }
        self.write_params();

        loop {
            let (frame, is_last_frame) = if let Some(decoder) = self.decoder.as_mut() {
                let (frame, is_last_frame) = decoder.read_frame()?;
                    (Some(frame), is_last_frame)
            }
            else { (None, false) };

            if self.py_needs_reload() {
                if let Err(err) = self.reload_py_config() {
                    warn!("{}", err);
                }
            }

            if (is_last_frame && self.encoder.is_some()) || window_exit_requested {
                break;
            }
            else if is_last_frame && self.decoder.as_ref().unwrap().num_frames > 1 {
                self.decoder = Some(Decoder::new(&self.args.input_file.as_ref().unwrap(), self.args.width, self.args.height).context("Failed to create decoder")?);
            }

            // Render to swapchain or file
            if self.encoder.is_some() {
                self.execute(frame.as_deref(), Some(&mut rf_output));

                // Encode to file
                self.encoder.as_mut().unwrap().write_frame(&rf_output)?;
            }
            else {
                window_exit_requested = self.execute(frame.as_deref(), None);
            }
        }
    
        Ok(())
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

    let mut reforge = Reforge::new(args)?;

    reforge.run()
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

    fn test_single_io_compute() -> Result<f64> {
        let func_name = std::thread::current().name().unwrap().to_string();
        let input = "tests/images/waterfall.jpg";
        let candidate = &format!("tests/candidate-images/{func_name}.png");
        let reference = &format!("tests/reference-images/{func_name}.png");

        let args = make_args(&format!("-i {input} --shader-file tests/shaders/passthrough.comp -o {candidate}"));
        let _ = run_reforge(args)?;

        compare_images(candidate, reference)
    }

    fn test_chaining_io_compute() -> Result<f64> {
        let func_name = std::thread::current().name().unwrap().to_string();
        let input = "tests/images/waterfall.jpg";
        let candidate = &format!("tests/candidate-images/{func_name}.png");
        let reference = &format!("tests/reference-images/{func_name}.png");

        let args = make_args(&format!("-i {input} --shader-path tests/shaders --py-config-path tests/py -p chaining_io -o {candidate}"));
        let _ = run_reforge(args)?;

        compare_images(candidate, reference)
    }

    fn test_compare_res(compare: Result<f64>) {
        if let Err(err) = &compare {
            eprintln!("{:?}", err);
        }
        assert!(compare.is_ok());
        assert!(compare.unwrap() > 0.95);
    }

    #[test]
    fn single_io_compute() {
        test_compare_res(test_single_io_compute())
    }

    #[test]
    fn chaining_io_compute() {
        test_compare_res(test_chaining_io_compute())
    }
}
