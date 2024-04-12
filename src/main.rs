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
use tracing::{debug, info};
use utils::TERM_CLEAR;

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

#[derive(clap::Parser)]
pub struct Args {
    #[arg(short='i', long="input-file", help = "File to read from")]
    input_file: Option<String>,

    #[arg(short='o', long="output-file", help = "Jpg file to write to")]
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
        let env_path = std::env::var("EOB1_SHADER_PATH").context("Missing required --shader-path, --shader-file or env var EOB1_SHADER_PATH")?;
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

fn main() -> Result<()> {
    let args = Args::parse();

    let log_level = args.log_level.unwrap_or(LogLevel::Warn).to_trace();

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(log_level).finish();

    tracing::subscriber::set_global_default(subscriber).context("setting tracing default failed")?;

    let span = tracing::trace_span!("reforge");
    let _guard = span.enter();

    let args = update_args(args)?;

    let use_swapchain = args.output_file.is_none();

    // Only one frame to be in flight if we aren't using the swapchain
    let num_frames = if use_swapchain { args.num_frames.unwrap() } else { 1 } ;

    if args.config.is_some() && args.shader_file.is_some() {
        warnln!("Cannot specify both a config and shader file");
        std::process::exit(1);
    }

    let (mut width, mut height) = utils::get_dim(800, 600, args.width, args.height);

    let mut decoder = if let Some(input_file) = &args.input_file {
        let decoder = Decoder::new(input_file, args.width, args.height).context("Failed to create decoder")?;
        info!("Decoded size: {} X {}", decoder.width, decoder.height);
        width = decoder.width;
        height = decoder.height;
        Some(decoder)
    } else { None };

    let encoder = if let Some(output_file) = &args.output_file {
        Some(Encoder::new(&output_file, width, height).context("Failed to create encoder")?)
    } else { None };

    let (graph, py_config_timestamp) = if let Some(python_config) = args.python_config.as_ref() {
        let graph = py::py_config(&python_config)?;
        (graph, utils::get_modified_time(&python_config))
    }
    else {
        // Verified by this point in an earlier check
        let shader_file = args.shader_file.as_ref().unwrap();

        let graph = match decoder {
            Some(_) => format!("input -> {} -> output", shader_file),
            None    => format!(         "{} -> output", shader_file)
        };

        (graph, 0)
    };

    let shader_path = args.shader_path.clone().unwrap_or("".to_string());
    let config = config::parse(graph.clone(), &shader_path).context("Failed to create config")?;
    debug!("Config: {:?}", config);

    let render_info = RenderInfo {
        config,
        width: width,
        height: height,
        num_frames: num_frames,
        format: args.shader_format.unwrap().to_vk_format(),
        swapchain: use_swapchain,
        has_input_image: args.input_file.is_some(),
    };

    let event_loop = if use_swapchain { Some(EventLoop::new()) } else { None };
    let mut render = Render::new(render_info, &event_loop);

    let mut first_run = vec![true; num_frames];

    unsafe {
    let mut avg_ms = 0.0;
    let mapped_input_image_data: *mut u8 = render.staging_buffer_ptr();

    let mut timer: std::time::Instant = std::time::Instant::now();
    let time_since_start: std::time::Instant = std::time::Instant::now();

    // Decode the file into the staging buffer
    let (frame, is_last_frame) = if let Some(decoder) = decoder.as_mut() {
        let (frame, is_last_frame) = decoder.read_frame()?;
        unsafe { std::ptr::copy_nonoverlapping(frame.as_ptr(), mapped_input_image_data, frame.len()); }
        (Some(frame), is_last_frame)
    }
    else { (None, false) };

    let elapsed_ms = utils::get_elapsed_ms(&timer);
    println!("File Decode and resize: {:.2}ms", elapsed_ms);

    let mut render_fn = |render: &mut Render| {
        // Wait for the previous iteration of this frame before
        // changing or executing on its resources
        render.wait_for_frame_fence();

        if render.trigger_reloads() {
            // Clear current line of timers
            eprint!("{TERM_CLEAR}");
            first_run.iter_mut().for_each(|b| *b = true);
        }

        render.update_ubos(time_since_start.elapsed().as_secs_f32());

        // Pull in the next image from the swapchain
        if use_swapchain {
            render.acquire_swapchain();
        }

        let elapsed_ms = utils::get_elapsed_ms(&timer);
        avg_ms = utils::moving_avg(avg_ms, elapsed_ms);
        timer = std::time::Instant::now();

        let gpu_times = render.last_frame_gpu_times();
        eprint!("\rFrame: {:5.2}ms, Frame-Avg: {:5.2}ms, GPU: {{{}}}", elapsed_ms, avg_ms, gpu_times);

        render.begin_record();

        // On the first run, we:
        // 1. Transitioned images as needed
        // 2. Load the staging input buffer into an image and convert it to linear
        if first_run[render.frame_index] {
            if args.input_file.is_some() {
                render.record_initial_image_load();
            }
            render.record_pipeline_image_transitions();
            first_run[render.frame_index] = false;
        }

        render.record();

        if !use_swapchain {
            render.write_output_to_buffer();
        }

        render.end_record();

        // Send the work to the gpu
        render.submit();
    };

    let mut first_resize = true;

    if use_swapchain {
        event_loop.unwrap().run_return(|event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::WindowEvent {
                    event: WindowEvent::Resized(_size),
                    ..
                } => {
                    // This event gets triggered on initial window creation
                    // and we don't want to recreate the swapchain at that point
                    if !first_resize {
                        render.swapchain_rebuilt_required = true;
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
                ..
                } => *control_flow = ControlFlow::Exit,
                Event::MainEventsCleared => render_fn(&mut render),
                _ => (),
            }

        });
        //render_loop(&mut event_loop, &mut render_fn);
    }
    else {
        render_fn(&mut render);
        render.wait_for_frame_fence();
        if let Some(mut encoder) = encoder {
            let slice = core::slice::from_raw_parts(mapped_input_image_data, (width as usize)*(height as usize)*4);
            encoder.write_frame(slice)?;
        }
    }

    }

    Ok(())
}
