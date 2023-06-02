extern crate ash;
extern crate shaderc;
extern crate clap;
extern crate gpu_allocator;

use gpu_allocator as gpu_alloc;
use gpu_allocator::vulkan as gpu_alloc_vk;

use clap::Parser;

use ash::vk;
use std::collections::HashMap;
use std::default::Default;
use std::rc::Rc;

mod vulkan;
use vulkan::command;
use vulkan::swapchain::SwapChain;
use vulkan::core::VkCore;
use vulkan::pipeline_graph::PipelineInfo;
use vulkan::pipeline_graph::PipelineGraph;
use vulkan::pipeline_graph::NUM_FRAMES;
use vulkan::frame::Frame;
use vulkan::utils;

mod imagefileio;
use imagefileio::ImageFileDecoder;

#[cfg(any(target_os = "macos", target_os = "ios"))]
use ash::vk::{
    KhrGetPhysicalDeviceProperties2Fn, KhrPortabilityEnumerationFn, KhrPortabilitySubsetFn,
};

#[derive(clap::Parser, Debug)]
pub struct Args {
    #[arg(value_name="input-file")]
    input_file: String,

    #[arg(long)]
    width: Option<u32>,

    #[arg(long)]
    height: Option<u32>
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

fn get_modified_times(pipeline_infos: &HashMap<&str, PipelineInfo>) -> HashMap<String, u64> {
    let mut timestamps: HashMap<String, u64> = HashMap::new();

    for (name, info) in pipeline_infos {
        let timestamp = std::fs::metadata(&info.shader_path).unwrap().modified().unwrap().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_secs();

        timestamps.insert(name.to_string(), timestamp);
    }

    timestamps
}

fn get_dim(width: u32, height: u32, new_width: Option<u32>, new_height: Option<u32>) -> (u32, u32) {
    let mut w  = width;
    let mut h = height;

    if new_width.is_some() && new_height.is_some() {
        return (new_width.unwrap(), new_height.unwrap())
    }

    if new_width.is_some() {
        w = new_width.unwrap();
        h = ((w as f32/width as f32)*height as f32) as u32;
    }
    else if new_height.is_some() {
        h = new_height.unwrap();
        w = ((h as f32/(height as f32))*width as f32) as u32;
    }

    (w, h)
}

fn moving_avg(mut avg: f64, next_value: f64) -> f64 {

    avg -= avg / 60.0;
    avg += next_value / 60.0;

    return avg;
}

fn get_elapsed_ms(inst: &std::time::Instant) -> f64{
    return (inst.elapsed().as_nanos() as f64)/1e6 as f64;
}

fn main() {
    let args = Args::parse();

    imagefileio::init();

    let mut file_decoder = ImageFileDecoder::new(&args.input_file);

    let (window_width, window_height) = get_dim(file_decoder.width, file_decoder.height, args.width, args.height);

    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Reforge")
        .with_inner_size(winit::dpi::PhysicalSize::new(
            f64::from(window_width),
            f64::from(window_height),
        ))
        .with_resizable(true)
        .build(&event_loop)
        .unwrap();

    unsafe {
    let vk_core = Rc::new(VkCore::new(&window));

    // setting up the allocator
    let mut allocator = gpu_alloc_vk::Allocator::new(&gpu_alloc_vk::AllocatorCreateDesc {
        instance: vk_core.instance.clone(),
        device: vk_core.device.clone(),
        physical_device: vk_core.pdevice,
        debug_settings: Default::default(),
        buffer_device_address: false,
    }).unwrap();

    let pipeline_infos = HashMap::from([
        ("contrast-pipeline", PipelineInfo {
            shader_path: "shaders/contrast.comp".to_string(),
            input_images: [(0, "file".to_string())].to_vec(),
            output_images: [(1, "contrast".to_string())].to_vec()
        }),

        ("contrast-pipeline2", PipelineInfo {
            shader_path: "shaders/contrast.comp".to_string(),
            input_images: [(0, "contrast".to_string())].to_vec(),
            output_images: [(1, "contrast2".to_string())].to_vec(),
        }),

        ("brightness-pipeline", PipelineInfo {
            shader_path: "shaders/brightness.comp".to_string(),
            input_images: [(0, "contrast2".to_string())].to_vec(),
            output_images: [(1, "swapchain".to_string())].to_vec(),
        })
    ]);

    let mut graph = PipelineGraph::new(Rc::clone(&vk_core), &mut allocator, &pipeline_infos, &window);
    let frames : Vec<Frame> = (0..NUM_FRAMES).map(|_|{
        Frame::new(&vk_core)
    }).collect();


    let buffer_size = (window_width as vk::DeviceSize)*(window_height as vk::DeviceSize)*4;
    let input_image_buffer = utils::create_buffer(&vk_core.device,
                                                  buffer_size,
                                                  vk::BufferUsageFlags::TRANSFER_SRC,
                                                  gpu_alloc::MemoryLocation::CpuToGpu,
                                                  &mut allocator);

    // Pipeline-name -> timestamp
    let mut last_modified_shader_times: HashMap<String, u64>  = get_modified_times(&pipeline_infos);

    let mapped_input_image_data: *mut u8 = input_image_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;
    let mut timer: std::time::Instant = std::time::Instant::now();

    file_decoder.decode(mapped_input_image_data, window_width, window_height);
    let elapsed_ms = get_elapsed_ms(&timer);
    println!("File Decode and resize: {:.2?}ms", elapsed_ms);

    let swapchain = SwapChain::new(&vk_core, window_width, window_height);

    let mut avg_ms = 0.0;

    render_loop(&mut event_loop, &mut || {
        static mut FIRST_RUN: [bool;NUM_FRAMES] = [true ; NUM_FRAMES];
        static mut FRAME_INDEX: usize = 0;


        let current_modified_shader_times: HashMap<String, u64>  = get_modified_times(&pipeline_infos);

        for (name, timestamp) in &last_modified_shader_times {
            if *current_modified_shader_times.get(name).unwrap() != *timestamp {
                graph.rebuild_pipeline(name);
            }
        }

        last_modified_shader_times = current_modified_shader_times;

        let graph_frame = &graph.frames[FRAME_INDEX];
        let frame = &frames[FRAME_INDEX];
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


        let elapsed_ms = get_elapsed_ms(&timer);
        avg_ms = moving_avg(avg_ms, elapsed_ms);
        print!("\rFrame: {:.2?}ms , Avg: {:.2?}ms ", elapsed_ms, avg_ms);
        timer = std::time::Instant::now();

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

        if FIRST_RUN[FRAME_INDEX] {
            if FRAME_INDEX == 0 {
                let regions = vk::BufferImageCopy {
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

                // Copy user-input image from vk buffer to the input image
                command::transition_image_layout(&device, frame.cmd_buffer, input_image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                device.cmd_copy_buffer_to_image(frame.cmd_buffer, input_image_buffer.vk, input_image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[regions]);
                command::transition_image_layout(&device, frame.cmd_buffer, input_image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL);
            }

            // Transition all intermediate compute images to general
            for (name, image) in &graph_frame.images {
                if name != "swapchain" && name != "file" {
                    command::transition_image_layout(&device, frame.cmd_buffer, image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
                }
            }

            FIRST_RUN[FRAME_INDEX] = false;
        }

        command::transition_image_layout(&device, frame.cmd_buffer, graph_frame.images.get("swapchain").unwrap().vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);

        command::execute_pipeline_graph(&device, frame, graph_frame, &graph);

        command::transition_image_layout(&device, frame.cmd_buffer, graph_frame.images.get("swapchain").unwrap().vk, vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        command::transition_image_layout(&device, frame.cmd_buffer, swapchain.images[present_index as usize], vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

        /* TODO?: Currently, we are using blit_image because it will do the format
         * conversion for us. However, another alternative is to do copy_image
         * after specifying th final compute shader destination image as the same
         * format as the swapchain format. Maybe worth measuring perf difference later */
        command::blit_copy(device, frame.cmd_buffer, &command::BlitCopy {
            width: window_width,
            height: window_height,
            src_image: graph_frame.images.get("swapchain").unwrap().vk,
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

        FRAME_INDEX = (FRAME_INDEX+1)%NUM_FRAMES;
    });

    }
}
