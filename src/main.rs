extern crate ash;
extern crate shaderc;
extern crate gpu_allocator;
extern crate clap;

use ash::vk::Offset3D;
use clap::Parser;
use gpu_allocator as gpu_alloc;

use ash::vk;
use std::default::Default;
use std::rc::Rc;

mod vulkan;
use vulkan::core::VkCore;
use vulkan::pipeline_factory::PipelineInfo;
use vulkan::pipeline_factory::PipelineFactory;
use vulkan::pipeline_factory::Image;
use vulkan::pipeline_factory::NUM_FRAMES;
use vulkan::pipeline_factory::SHADER_PATH;

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

pub struct VkSwapRes {
    pub descriptor_set: ash::vk::DescriptorSet,
}

unsafe fn create_resizable_res(core: &VkCore,
                               res: &PipelineFactory,
                               input_image: &Image) -> (Vec<VkSwapRes>, vk::DescriptorPool)
{
    let descriptor_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            // Input + output, so 2*
            descriptor_count: 2*res.swapchain.images.len() as u32,
        },
    ];
    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&descriptor_sizes)
        .max_sets(res.swapchain.images.len() as u32);

    let descriptor_pool = core.device
        .create_descriptor_pool(&descriptor_pool_info, None)
        .unwrap();

    let desc_layout = &[res.pipeline_layout.descriptor_layout];
    let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(desc_layout);

    let swap_res : Vec<VkSwapRes> = (0..res.swapchain.images.len()).map(|i|{
        let descriptor_set = core.device
            .allocate_descriptor_sets(&desc_alloc_info)
            .unwrap()[0];

        let input_image_descriptor = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: input_image.view,
            ..Default::default()
        };

        let output_image_descriptor = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: res.swapchain.views[i],
            ..Default::default()
        };

        let write_desc_sets = [
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                p_image_info: &input_image_descriptor,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                p_image_info: &output_image_descriptor,
                ..Default::default()
            },
        ];
        core.device.update_descriptor_sets(&write_desc_sets, &[]);

        VkSwapRes {
            descriptor_set : descriptor_set
        }
    }).collect();

    (swap_res, descriptor_pool)
}



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

fn get_modified_time(path: &str) -> u64 {
    std::fs::metadata(path).unwrap().modified().unwrap().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_secs()
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

    let info = PipelineInfo {
        shader_path: "shaders/shader.comp".to_string(),
        input_images: [(0, "file".to_string())].to_vec(),
        output_images: [(1, "swapchain".to_string())].to_vec(),
    };

    res.add("test-pipeline", info);
    res.build();


    let buffer_size = (window_width as vk::DeviceSize)*(window_height as vk::DeviceSize)*4;
    let input_image_buffer = res.create_buffer(buffer_size, vk::BufferUsageFlags::TRANSFER_SRC, gpu_alloc::MemoryLocation::CpuToGpu);

    //let input_image = create_input_image(&device, window_size.width, window_size.height, &mut allocator);
    let (swap_res, descriptor_pool) = {
        res.create_image("input".to_string(), window_width, window_height);
        let input_image = res.get_image("input".to_string());
        create_resizable_res(&vk_core, &res, &input_image)
    };

    let mut last_modified_shader_time = get_modified_time(SHADER_PATH);

    let mapped_input_image_data: *mut u8 = input_image_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8;

    file_decoder.decode(mapped_input_image_data, window_width, window_height);

    render_loop(&mut event_loop, &mut || {
        static mut FIRST_RUN: bool = true;
        static mut FRAME_INDEX: u8 = 0;

        let current_modified_time = get_modified_time(SHADER_PATH);

        if current_modified_time != last_modified_shader_time
        {
            res.rebuild_changed_compute_pipeline();
        }

        last_modified_shader_time = current_modified_time;

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

        let swap_res = &swap_res[present_index as usize];

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

            let image_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image: input_image.vk,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };

            device.cmd_pipeline_barrier(frame.cmd_buffer,
                                        vk::PipelineStageFlags::TOP_OF_PIPE,
                                        vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_barrier]);

            device.cmd_copy_buffer_to_image(frame.cmd_buffer, input_image_buffer.vk, input_image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[regions]);

            let image_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::GENERAL,
                image: input_image.vk,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                        },
                ..Default::default()
            };

            device.cmd_pipeline_barrier(frame.cmd_buffer,
                                        vk::PipelineStageFlags::TRANSFER,
                                        vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &[image_barrier]);

            FIRST_RUN = false;
        }

        let image_barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::SHADER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::GENERAL,
            //image: res.swapchain.images[present_index as usize],
            image: frame.images.get("swapchain").unwrap().vk,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
                    },
            ..Default::default()
        };

        device.cmd_pipeline_barrier(frame.cmd_buffer,
                                    vk::PipelineStageFlags::TOP_OF_PIPE,
                                    vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &[image_barrier]);

        let dispatch_x = (window_width as f32/16.0).ceil() as u32;
        let dispatch_y = (window_height as f32/16.0).ceil() as u32;

        device.cmd_bind_descriptor_sets(
            frame.cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            //res.pipeline_layout.vk,
            res.pipelines.get("test-pipeline").unwrap().layout.vk,
            0,
            //&[swap_res.descriptor_set],
            &[frame.descriptor_set],
            &[],
        );
        device.cmd_bind_pipeline(
            frame.cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            //res.compute_pipeline,
            res.pipelines.get("test-pipeline").unwrap().vk_pipeline
        );
        device.cmd_dispatch(frame.cmd_buffer, dispatch_x, dispatch_y, 1);


        let image_barrier_swap_transfer = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::GENERAL,
            new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image: frame.images.get("swapchain").unwrap().vk,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
                    },
            ..Default::default()
        };

        device.cmd_pipeline_barrier(frame.cmd_buffer,
                                    vk::PipelineStageFlags::COMPUTE_SHADER,
                                    vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_barrier_swap_transfer]);


        let image_barrier_transfer = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::NONE,
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            image: res.swapchain.images[present_index as usize],
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
                    },
            ..Default::default()
        };

        device.cmd_pipeline_barrier(frame.cmd_buffer,
                                    vk::PipelineStageFlags::TOP_OF_PIPE,
                                    vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_barrier_transfer]);

        let copy_subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1
        };

        let begin_offset = Offset3D {
            x: 0, y: 0, z: 0
        };
        let end_offset = Offset3D {
            x: window_width as i32, y: window_height as i32, z: 1
        };

        let blit = vk::ImageBlit {
            src_subresource: copy_subresource,
            src_offsets: [begin_offset, end_offset],
            dst_subresource: copy_subresource,
            dst_offsets: [begin_offset, end_offset]
        };

        /* TODO?: Currently, we are using blit_image because it will do the format
         * conversion for us. However, another alternative is to do copy_image
         * after specifying th final compute shader destination image as the same
         * format as the swapchain format. Maybe worth measuring perf difference later */
        device.cmd_blit_image(frame.cmd_buffer,
                              frame.images.get("swapchain").unwrap().vk,
                              vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                              res.swapchain.images[present_index as usize],
                              vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                              &[blit],
                              vk::Filter::LINEAR);

        let image_barrier_present = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            image: res.swapchain.images[present_index as usize],
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
                    },
            ..Default::default()
        };

        device.cmd_pipeline_barrier(frame.cmd_buffer,
                                    vk::PipelineStageFlags::TRANSFER,
                                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::DependencyFlags::empty(), &[], &[], &[image_barrier_present]);


        device.end_command_buffer(frame.cmd_buffer);


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
