extern crate ash;
extern crate clap;
extern crate gpu_allocator;
extern crate shaderc;
extern crate ffmpeg_sys_next as ffmpeg;
#[macro_use] extern crate lalrpop_util;

mod config;
mod imagefileio;
mod render;
mod utils;
mod vulkan;

use ash::vk;
use clap::Parser;
use imagefileio::ImageFileDecoder;
use imagefileio::ImageFileEncoder;
use render::Render;
use render::RenderInfo;
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

#[derive(clap::Parser)]
pub struct Args {
    #[arg(value_name="input-file", help = "Optional file to read from")]
    input_file: Option<String>,

    #[arg(value_name="output-file", help = "Optional jpg file to write to")]
    output_file: Option<String>,

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
}

fn render_loop<F: FnMut()>(event_loop: &mut EventLoop<()>, f: &mut F) {
    event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
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
            Event::MainEventsCleared => f(),
            _ => (),
        }
    });
}

fn main() {
    let args = Args::parse();
    let use_swapchain = args.output_file.is_none();

    // Only one frame to be in flight if we aren't using the swapchain
    let num_frames = if use_swapchain { args.num_frames.unwrap() } else { 1 } ;

    imagefileio::init();

    let file_decoder = match args.input_file.as_ref() {
        Some(input_file) => {
            match ImageFileDecoder::new(&input_file) {
                Ok(decoder) => Some(decoder),
                Err(err) => panic!("{}", err)
            }
        },
        None => None
    };

    let (width, height) = match file_decoder.as_ref() {
        Some(decoder) => utils::get_dim(decoder.width, decoder.height, args.width, args.height),
        None => utils::get_dim(800, 600, args.width, args.height)
    };

    let render_info = RenderInfo {
        width: width,
        height: height,
        num_frames: num_frames,
        config_path: args.config,
        format: args.shader_format.unwrap().to_vk_format(),
        swapchain: use_swapchain,
        has_input_image: args.input_file.is_some()
    };

    let mut event_loop = EventLoop::new();
    let mut render = Render::new(render_info, &event_loop);

    let mut first_run = vec![true; num_frames];

    unsafe {
    let mut avg_ms = 0.0;
    let mapped_input_image_data: *mut u8 = render.staging_buffer_ptr();

    let mut timer: std::time::Instant = std::time::Instant::now();
    let time_since_start: std::time::Instant = std::time::Instant::now();

    // Decode the file into the staging buffer
    if file_decoder.as_ref().is_some() {
        file_decoder.unwrap().decode(mapped_input_image_data, width, height).unwrap_or_else(|err| panic!("Error: {}", err));
    }

    let elapsed_ms = utils::get_elapsed_ms(&timer);
    println!("File Decode and resize: {:.2}ms", elapsed_ms);

    let mut render_fn = || {
        // Wait for the previous iteration of this frame before
        // changing or executing on its resources
        render.wait_for_frame_fence();

        // If our configuration has changed, live reload it
        if render.reload_changed_config().is_some() {
            first_run.iter_mut().for_each(|b| *b = true);
            // Clear current line of timers
            eprint!("{TERM_CLEAR}");
        }

        // If any of our shaders have changed, live reload them
        render.reload_changed_pipelines();

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

    if use_swapchain {
        render_loop(&mut event_loop, &mut render_fn);
    }
    else {
        render_fn();
        render.wait_for_frame_fence();
        ImageFileEncoder::encode(&args.output_file.unwrap(), mapped_input_image_data, width as i32, height as i32).unwrap_or_else(|err| panic!("Encoding error: {}", err));
    }

    }
}
