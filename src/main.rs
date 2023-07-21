extern crate ash;
extern crate clap;
extern crate gpu_allocator;
extern crate shaderc;
#[macro_use] extern crate lalrpop_util;

use gpu_allocator as gpu_alloc;

mod config;
mod imagefileio;
mod render;
mod utils;
mod vulkan;

use ash::vk;
use clap::Parser;
use imagefileio::ImageFileDecoder;
use render::Render;
use render::RenderInfo;
use vulkan::vkutils;

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
    #[arg(value_name="input-file")]
    input_file: String,

    #[arg(long)]
    width: Option<u32>,

    #[arg(long)]
    height: Option<u32>,

    #[arg(long, default_value = "rgba32f", help = "Shader image format")]
    shader_format: Option<ShaderFormat>,

    #[arg(long, value_name="node-config", help = "Path to the node configuration file")]
    node_config: Option<String>,

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
    let num_frames = args.num_frames.unwrap();

    imagefileio::init();

    let mut file_decoder = ImageFileDecoder::new(&args.input_file);

    let (width, height) = utils::get_dim(file_decoder.width, file_decoder.height, args.width, args.height);

    let render_info = RenderInfo {
        width: width,
        height: height,
        num_frames: num_frames,
        config_path: args.node_config,
        format: args.shader_format.unwrap().to_vk_format()
    };

    let mut event_loop = EventLoop::new();
    let mut render = Render::new(render_info, &event_loop);

    let mut first_run = vec![true; num_frames];

    // We use rgba8 as the input file format
    let buffer_size = (width as vk::DeviceSize)*(height as vk::DeviceSize)*4;

    unsafe {

    // This staging buffer will be used to transfer the original input file into an mimage
    let input_image_buffer = vkutils::create_buffer(&render.vk_core,
                                                    "input-image-staging-buffer".to_string(),
                                                    buffer_size,
                                                    vk::BufferUsageFlags::TRANSFER_SRC,
                                                    gpu_alloc::MemoryLocation::CpuToGpu);

    let mut avg_ms = 0.0;
    let mapped_input_image_data: *mut u8 = input_image_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;

    let mut timer: std::time::Instant = std::time::Instant::now();
    let time_since_start: std::time::Instant = std::time::Instant::now();

    // Decode the file into the staging buffer
    file_decoder.decode(mapped_input_image_data, width, height);

    let elapsed_ms = utils::get_elapsed_ms(&timer);
    println!("File Decode and resize: {:.2?}ms", elapsed_ms);

    render_loop(&mut event_loop, &mut || {
        // Wait for the previous iteration of this frame before
        // changing or executing on its resources
        render.wait_for_frame_fence();

        // If our configuration has changed, live reload it
        if render.reload_changed_config() {
            first_run.iter_mut().for_each(|b| *b = true);
        }

        // If any of our shaders have changed, live reload them
        render.reload_changed_pipelines();

        render.update_ubos(time_since_start.elapsed().as_secs_f32());

        // Pull in the next image from the swapchain
        render.acquire_swapchain();

        let elapsed_ms = utils::get_elapsed_ms(&timer);
        avg_ms = utils::moving_avg(avg_ms, elapsed_ms);
        timer = std::time::Instant::now();

        let gpu_times = render.last_frame_gpu_times();
        print!("\rFrame: {:.2?}ms , Frame-Avg: {:.2?}ms, GPU: {{{}}}", elapsed_ms, avg_ms, gpu_times);

        render.begin_record();

        // On the first run, we:
        // 1. Transitioned images as needed
        // 2. Load the staging input buffer into an image and convert it to linear
        if first_run[render.frame_index] {
            render.record_initial_image_load(&input_image_buffer);
            render.record_pipeline_image_transitions();
            first_run[render.frame_index] = false;
        }

        render.record();
        render.end_record();

        // Send the work to the gpu
        render.submit();
    });

    }
}
