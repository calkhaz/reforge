extern crate ash;
extern crate shaderc;
extern crate gpu_allocator;
extern crate clap;

use clap::Parser;
use gpu_allocator as gpu_alloc;

use ash::vk;
use std::collections::HashMap;
use std::default::Default;
use std::rc::Rc;

mod vulkan;
use vulkan::command;
use vulkan::core::VkCore;
use vulkan::pipeline_factory::PipelineInfo;
use vulkan::pipeline_factory::PipelineFactory;
use vulkan::pipeline_factory::NUM_FRAMES;

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

fn get_modified_times(pipeline_infos: &HashMap<String, PipelineInfo>) -> HashMap<String, u64> {
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
    let mut res = PipelineFactory::new(Rc::clone(&vk_core), &window);

    res.add("contrast-pipeline", PipelineInfo {
        shader_path: "shaders/contrast.comp".to_string(),
        input_images: [(0, "file".to_string())].to_vec(),
        output_images: [(1, "contrast".to_string())].to_vec(),
    });

    res.add("brightness-pipeline", PipelineInfo {
        shader_path: "shaders/brightness.comp".to_string(),
        input_images: [(0, "contrast".to_string())].to_vec(),
        output_images: [(1, "swapchain".to_string())].to_vec(),
    });

    res.build();

    let buffer_size = (window_width as vk::DeviceSize)*(window_height as vk::DeviceSize)*4;
    let input_image_buffer = res.create_buffer(buffer_size, vk::BufferUsageFlags::TRANSFER_SRC, gpu_alloc::MemoryLocation::CpuToGpu);

    // Pipeline-name -> timestamp
    let mut last_modified_shader_times: HashMap<String, u64>  = get_modified_times(&res.pipeline_infos);

    let mapped_input_image_data: *mut u8 = input_image_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;

    file_decoder.decode(mapped_input_image_data, window_width, window_height);

    render_loop(&mut event_loop, &mut || {
        static mut FIRST_RUN: bool = true;
        static mut FRAME_INDEX: u8 = 0;

        let current_modified_shader_times: HashMap<String, u64>  = get_modified_times(&res.pipeline_infos);

        for (name, timestamp) in &last_modified_shader_times {
            if *current_modified_shader_times.get(name).unwrap() != *timestamp {
                res.rebuild_pipeline(name);
            }
        }

        last_modified_shader_times = current_modified_shader_times;

        let frame = &res.frames[FRAME_INDEX as usize];
        let device = &vk_core.device;

        let (present_index, _) = res
            .swapchain.loader.acquire_next_image(
                res.swapchain.vk,
                std::u64::MAX,
                frame.present_complete_semaphore, // Semaphore to signal
                vk::Fence::null(),
            )
            .unwrap();

        device
            .wait_for_fences(&[frame.fence], true, std::u64::MAX)
            .expect("Wait for fence failed.");

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



        if FIRST_RUN {
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

            let input_image = &res.get_input_image();

            // Copy user-input image from vk buffer to the input image
            command::transition_image_layout(&device, frame.cmd_buffer, input_image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            device.cmd_copy_buffer_to_image(frame.cmd_buffer, input_image_buffer.vk, input_image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[regions]);
            command::transition_image_layout(&device, frame.cmd_buffer, input_image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL);


            FIRST_RUN = false;
        }

        command::transition_image_layout(&device, frame.cmd_buffer, frame.images.get("contrast").unwrap().vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
        command::transition_image_layout(&device, frame.cmd_buffer, frame.images.get("swapchain").unwrap().vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);

        let dispatch_x = (window_width as f32/16.0).ceil() as u32;
        let dispatch_y = (window_height as f32/16.0).ceil() as u32;

        device.cmd_bind_descriptor_sets(
            frame.cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            res.pipelines.get("contrast-pipeline").unwrap().layout.vk,
            0,
            &[*frame.descriptor_sets.get("contrast-pipeline").unwrap()],
            &[],
        );
        device.cmd_bind_pipeline(
            frame.cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            res.pipelines.get("contrast-pipeline").unwrap().vk_pipeline
        );
        device.cmd_dispatch(frame.cmd_buffer, dispatch_x, dispatch_y, 1);


        let mem_barrier = vk::MemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_READ,
            dst_access_mask: vk::AccessFlags::SHADER_WRITE,
            ..Default::default()
        };

        device.cmd_pipeline_barrier(frame.cmd_buffer,
                                    vk::PipelineStageFlags::COMPUTE_SHADER,
                                    vk::PipelineStageFlags::COMPUTE_SHADER,
                                    vk::DependencyFlags::empty(), &[mem_barrier], &[], &[]);


        device.cmd_bind_descriptor_sets(
            frame.cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            res.pipelines.get("brightness-pipeline").unwrap().layout.vk,
            0,
            &[*frame.descriptor_sets.get("brightness-pipeline").unwrap()],
            &[],
        );
        device.cmd_bind_pipeline(
            frame.cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            res.pipelines.get("brightness-pipeline").unwrap().vk_pipeline
        );
        device.cmd_dispatch(frame.cmd_buffer, dispatch_x, dispatch_y, 1);



        command::transition_image_layout(&device, frame.cmd_buffer, frame.images.get("swapchain").unwrap().vk, vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

        command::transition_image_layout(&device, frame.cmd_buffer, res.swapchain.images[present_index as usize], vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

        /* TODO?: Currently, we are using blit_image because it will do the format
         * conversion for us. However, another alternative is to do copy_image
         * after specifying th final compute shader destination image as the same
         * format as the swapchain format. Maybe worth measuring perf difference later */
        command::blit_copy(device, frame.cmd_buffer, &command::BlitCopy {
            width: window_width,
            height: window_height,
            src_image: frame.images.get("swapchain").unwrap().vk,
            dst_image: res.swapchain.images[present_index as usize],
            src_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL
        });

        command::transition_image_layout(&device, frame.cmd_buffer, res.swapchain.images[present_index as usize], vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);

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
        let swapchains = [res.swapchain.vk];
        let image_indices = [present_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        res.swapchain.loader
            .queue_present(vk_core.queue, &present_info)
            .unwrap();

        FRAME_INDEX = (FRAME_INDEX+1)%NUM_FRAMES;
    });

    }
}
