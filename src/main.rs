extern crate ash;
extern crate shaderc;
extern crate clap;
extern crate gpu_allocator;

use gpu_allocator as gpu_alloc;

use clap::Parser;

use ash::vk;
use std::collections::HashMap;
use std::default::Default;

mod vulkan;
use vulkan::command;
use vulkan::swapchain::SwapChain;
use vulkan::core::VkCore;
use vulkan::pipeline_graph::PipelineInfo;
use vulkan::pipeline_graph::PipelineGraph;
use vulkan::pipeline_graph::FILE_INPUT;
use vulkan::pipeline_graph::SWAPCHAIN_OUTPUT;
use vulkan::frame::Frame;
use vulkan::vkutils;

mod utils;

mod imagefileio;
use imagefileio::ImageFileDecoder;

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

    #[arg(long, default_value = "rgba8", help = "Shader image format")]
    shader_format: Option<ShaderFormat>,

    #[arg(long, default_value= "2", help = "Number of frame-in-flight to be used when displaying to the swapchain")]
    num_frames: Option<usize>,
}

use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::WindowBuilder,
};


fn render_loop<F: FnMut()>(event_loop: &mut EventLoop<()>, f: &mut F) {
    event_loop
        .run_return(|event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::WindowEvent {
                    event:
                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
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

    let (window_width, window_height) = utils::get_dim(file_decoder.width, file_decoder.height, args.width, args.height);

    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Reforge")
        .with_inner_size(winit::dpi::PhysicalSize::new(
            f64::from(window_width),
            f64::from(window_height),
        ))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    unsafe {
    let vk_core = VkCore::new(&window);

    // Example of chained shaders (currently all just passthrough)
    let pipeline_infos = HashMap::from([
        ("gaussian-h", PipelineInfo {
            shader_path: "shaders/gaussian-horizontal.comp".to_string(),
            input_images: [(0, FILE_INPUT.to_string())].to_vec(),
            output_images: [(1, "gauss-h".to_string())].to_vec()
        }),
        ("gaussian-v", PipelineInfo {
            shader_path: "shaders/gaussian-vertical.comp".to_string(),
            input_images: [(0, "gauss-h".to_string())].to_vec(),
            output_images: [(1, SWAPCHAIN_OUTPUT.to_string())].to_vec(),
        })
    ]);

    /*
    let pipeline_infos = HashMap::from([
        ("passthrough-pipeline", PipelineInfo {
            shader_path: "shaders/passthrough.comp".to_string(),
            input_images: [(0, FILE_INPUT.to_string())].to_vec(),
            output_images: [(1, SWAPCHAIN_OUTPUT.to_string())].to_vec(),
        })
    ]);
    */

    let mut graph = PipelineGraph::new(&vk_core, &pipeline_infos, args.shader_format.unwrap().to_vk_format(), window_width, window_height, num_frames);
    let mut frames : Vec<Frame> = (0..num_frames).map(|_|{
        Frame::new(&vk_core, pipeline_infos.len() as u32)
    }).collect();


    let buffer_size = (window_width as vk::DeviceSize)*(window_height as vk::DeviceSize)*4;
    let input_image_buffer = vkutils::create_buffer(&vk_core,
                                                    "input-image-staging-buffer".to_string(),
                                                    buffer_size,
                                                    vk::BufferUsageFlags::TRANSFER_SRC,
                                                    gpu_alloc::MemoryLocation::CpuToGpu);
    let input_srgb_image = vkutils::create_image(&vk_core,
                                                 "input-image-srgb".to_string(),
                                                 vk::Format::R8G8B8A8_SRGB,
                                                 window_width, window_height);

    // Pipeline-name -> timestamp
    let mut last_modified_shader_times: HashMap<String, u64> = utils::get_modified_times(&pipeline_infos);

    let mapped_input_image_data: *mut u8 = input_image_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
    let mut timer: std::time::Instant = std::time::Instant::now();

    file_decoder.decode(mapped_input_image_data, window_width, window_height);
    let elapsed_ms = utils::get_elapsed_ms(&timer);
    println!("File Decode and resize: {:.2?}ms", elapsed_ms);

    let swapchain = SwapChain::new(&vk_core, window_width, window_height);

    let mut avg_ms = 0.0;

    let mut first_run = vec![true; num_frames];

    render_loop(&mut event_loop, &mut || {
        static mut FRAME_INDEX: usize = 0;

        let current_modified_shader_times: HashMap<String, u64> = utils::get_modified_times(&pipeline_infos);

        for (name, last_timestamp) in &last_modified_shader_times {
            match *current_modified_shader_times.get(name).unwrap() {
                // If the file was set to 0, we were unable to find it
                // Ex: File was moved or not available, print an error just once if we previously saw it
                0 => {
                    if 0 != *last_timestamp {
                        eprintln!("Unable to access shader file: {}", pipeline_infos.get(name.as_str()).unwrap().shader_path);
                    }
                }
                modified_timestamp => {
                    if modified_timestamp != *last_timestamp {
                        graph.rebuild_pipeline(&name);
                    }
                }
            }
        }

        last_modified_shader_times = current_modified_shader_times;

        let graph_frame = &graph.frames[FRAME_INDEX];
        let frame = &mut frames[FRAME_INDEX];
        let device = &vk_core.device;

        let (present_index, _) = swapchain.loader.acquire_next_image(
                swapchain.vk,
                std::u64::MAX,
                frame.present_complete_semaphore, // Semaphore to signal
                vk::Fence::null(),
            )
            .unwrap();

        device
            .wait_for_fences(&[frame.fence], true, std::u64::MAX)
            .expect("Wait for fence failed.");

        let elapsed_ms = utils::get_elapsed_ms(&timer);
        avg_ms = utils::moving_avg(avg_ms, elapsed_ms);
        timer = std::time::Instant::now();

        let gpu_times = frame.timer.get_elapsed_ms();
        print!("\rFrame: {:.2?}ms , Frame-Avg: {:.2?}ms, GPU: {{{}}}", elapsed_ms, avg_ms, gpu_times);

        device
            .reset_fences(&[frame.fence])
            .expect("Reset fences failed.");

        device
            .reset_command_buffer(
                frame.cmd_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .expect("Reset command buffer failed.");



        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();

        device.begin_command_buffer(frame.cmd_buffer, &command_buffer_begin_info)
            .expect("Begin commandbuffer");

        device.cmd_reset_query_pool(frame.cmd_buffer,
                                    frame.timer.query_pool,
                                    0, // first-query-idx
                                    frame.timer.query_pool_size);

        if first_run[FRAME_INDEX] {
            if FRAME_INDEX == 0 {
                let buffer_regions = vk::BufferImageCopy {
                    buffer_offset: input_image_buffer.allocation.offset(),
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        layer_count: 1,
                        ..Default::default()
                    },
                    image_extent: vk::Extent3D {
                        width: window_width as u32,
                        height: window_height as u32,
                        depth: 1
                    },
                    ..Default::default()
                };

                let input_image = &graph.get_input_image();

                /* The goal here is to copy the input file from a vulkan buffer to an srgb image
                 * "input_srgb_image" and then to a linear rgb "input_image" so we have the correct
                 * gamma */

                // 1. Transition the two input images so they are ready to be transfer destinations
                command::transition_image_layout(&device, frame.cmd_buffer, input_image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                command::transition_image_layout(&device, frame.cmd_buffer, input_srgb_image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

                // 2. Copy the buffer to the srgb image and then make it ready to transfer out
                device.cmd_copy_buffer_to_image(frame.cmd_buffer, input_image_buffer.vk, input_srgb_image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[buffer_regions]);
                command::transition_image_layout(&device, frame.cmd_buffer, input_srgb_image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

                // 3. Copy the srgb image to the linear input image and get it ready for general compute
                //    Note: If a regular image_copy is used here, we will not get the desired gamma
                //    correction from the format change
                command::blit_copy(&device, frame.cmd_buffer, &command::BlitCopy {
                    src_image: input_srgb_image.vk, src_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_image: input_image.vk,      dst_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    width: window_width,
                    height: window_height
                });
                command::transition_image_layout(&device, frame.cmd_buffer, input_image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL);
            }

            // Transition all intermediate compute images to general
            for (name, image) in &graph_frame.images {
                if name != SWAPCHAIN_OUTPUT && name != FILE_INPUT {
                    command::transition_image_layout(&device, frame.cmd_buffer, image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
                }
            }

            first_run[FRAME_INDEX] = false;
        }

        command::transition_image_layout(&device, frame.cmd_buffer, graph_frame.images.get(SWAPCHAIN_OUTPUT).unwrap().vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);

        command::execute_pipeline_graph(&device, frame, graph_frame, &graph);

        command::transition_image_layout(&device, frame.cmd_buffer, graph_frame.images.get(SWAPCHAIN_OUTPUT).unwrap().vk, vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        command::transition_image_layout(&device, frame.cmd_buffer, swapchain.images[present_index as usize], vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

        /* TODO?: Currently, we are using blit_image because it will do the format
         * conversion for us. However, another alternative is to do copy_image
         * after specifying th final compute shader destination image as the same
         * format as the swapchain format. Maybe worth measuring perf difference later */
        command::blit_copy(device, frame.cmd_buffer, &command::BlitCopy {
            width: window_width,
            height: window_height,
            src_image: graph_frame.images.get(SWAPCHAIN_OUTPUT).unwrap().vk,
            dst_image: swapchain.images[present_index as usize],
            src_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL
        });

        command::transition_image_layout(&device, frame.cmd_buffer, swapchain.images[present_index as usize], vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);

        device.end_command_buffer(frame.cmd_buffer).unwrap();

        let present_complete_semaphore = &[frame.present_complete_semaphore];
        let cmd_buffers = &[frame.cmd_buffer];
        let signal_semaphores = &[frame.render_complete_semaphore];

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(present_complete_semaphore)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COMPUTE_SHADER])
            .command_buffers(cmd_buffers)
            .signal_semaphores(signal_semaphores);

        vk_core.device.queue_submit(
            vk_core.queue,
            &[submit_info.build()],
            frame.fence,
        )
        .expect("queue submit failed.");

        let wait_semaphores = [frame.render_complete_semaphore];
        let swapchains = [swapchain.vk];
        let image_indices = [present_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        swapchain.loader
            .queue_present(vk_core.queue, &present_info)
            .unwrap();

        FRAME_INDEX = (FRAME_INDEX+1)%num_frames;
    });

    }
}
