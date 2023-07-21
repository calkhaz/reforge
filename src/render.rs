use ash::vk;

use crate::config::config::ConfigPipeline;
use crate::config::config::parse as config_parse;
use crate::utils;
use crate::vulkan::command;
use crate::vulkan::core::VkCore;
use crate::vulkan::frame::Frame;
use crate::vulkan::pipeline_graph::FILE_INPUT;
use crate::vulkan::pipeline_graph::PipelineGraph;
use crate::vulkan::pipeline_graph::PipelineGraphInfo;
use crate::vulkan::pipeline_graph::SWAPCHAIN_OUTPUT;
use crate::vulkan::swapchain::SwapChain;
use crate::vulkan::vkutils;
use crate::vulkan::vkutils::Buffer;
use crate::vulkan::vkutils::Image;

use std::collections::HashMap;
use std::default::Default;
use std::rc::Rc;

use winit::{
    event_loop::EventLoop,
    window::WindowBuilder,
};

pub struct RenderInfo {
    pub width: u32,
    pub height: u32,
    pub num_frames: usize,
    pub config_path: Option<String>,
    pub format: vk::Format,
}

pub struct Render {
    frames: Vec<Frame>,
    graph: PipelineGraph,
    info: RenderInfo,
    input_srgb_image: Image,
    last_modified_config_time: u64,
    last_modified_shader_times: HashMap<String, u64>,
    present_index: u32,
    pub frame_index: usize,
    swapchain: SwapChain,
    _window: winit::window::Window,
    pub vk_core: VkCore,
}

impl Render {
    fn create_window(event_loop: &EventLoop<()>, width: u32, height:u32) -> winit::window::Window {
        WindowBuilder::new()
            .with_title("Reforge")
            .with_inner_size(winit::dpi::PhysicalSize::new(
                f64::from(width),
                f64::from(height),
            ))
            .with_resizable(false)
            .build(&event_loop)
            .unwrap()
    }

    fn load_config(config_path: &Option<String>) -> Option<HashMap<String, ConfigPipeline>> {
        let node_config = if config_path.is_some() {
            match std::fs::read_to_string(config_path.clone().unwrap()) {
                Ok(contents) => contents,
                Err(e) => { eprintln!("Error reading file '{}' : {}", config_path.as_ref().unwrap(), e); std::process::exit(1); }
            }
        }
        else {
            // Default configuration
            "input -> passthrough -> output".to_string()
        };

        config_parse(node_config.to_string())
    }

    unsafe fn create_graph(vk_core: &VkCore, info: &RenderInfo, pipeline_config: &HashMap<String, ConfigPipeline>) -> Option<PipelineGraph> {
        let pipeline_infos = vkutils::synthesize_config(Rc::clone(&vk_core.device), &pipeline_config)?;

        let graph_info = PipelineGraphInfo {
            pipeline_infos: pipeline_infos,
            format: info.format,
            width: info.width,
            height: info.height,
            num_frames: info.num_frames
        };

        PipelineGraph::new(&vk_core, graph_info)
    }

    pub fn reload_changed_config(&mut self) -> bool {
        if self.info.config_path.is_none() {
            return false;
        }
        let config_path = self.info.config_path.as_ref().unwrap();

        let current_modified_config_time = utils::get_modified_time(&config_path);

        match current_modified_config_time {
            0 => {
                if 0 != self.last_modified_config_time {
                    eprintln!("Unable to access config file: {}", config_path);
                }
            },
            modified_timestamp => {
                if modified_timestamp == self.last_modified_config_time {
                    return false;
                }

                self.last_modified_config_time = current_modified_config_time;

                let config_contents = match std::fs::read_to_string(config_path.clone()) {
                    Ok(contents) => Some(contents),
                    Err(e) => { eprintln!("Error reading file '{}' : {}", config_path, e); None }
                };

                if config_contents.is_none() {
                    return false;
                }

                let pipeline_config = config_parse(config_contents.unwrap());

                if pipeline_config.is_none() {
                    return false;
                }

                unsafe {
                self.vk_core.device.device_wait_idle().unwrap();

                let num_pipelines = pipeline_config.as_ref().unwrap().len() as u32;
                let graph = Self::create_graph(&self.vk_core, &self.info, &pipeline_config.unwrap());

                if graph.is_none() {
                    return false;
                }

                self.graph = graph.unwrap();
                self.frames.iter_mut().for_each(|f| f.rebuild_timer(num_pipelines));
                }
                self.frame_index = 0;
                self.last_modified_shader_times = utils::get_modified_times(&self.graph.pipelines);

                return true;
            }
        };

        false
    }

    pub fn update_ubos(&mut self, time: f32) {
        self.graph.frames[self.frame_index].ubos.iter_mut().for_each(|(_pipeline_name, buffer_block_map)| {
            buffer_block_map.iter_mut().for_each(|(buffer_member_name, buffer_block)| {
                if buffer_member_name.ends_with("_rf_time") {
                    unsafe {
                    let ptr = buffer_block.buffer.mapped_data.offset(buffer_block.offset as isize) as *mut f32;
                    std::ptr::copy_nonoverlapping(&time, ptr, 1)
                    }
                }
            });
        });
    }

    pub fn reload_changed_pipelines(&mut self) {
        let current_modified_shader_times: HashMap<String, u64> = utils::get_modified_times(&self.graph.pipelines);

        for (name, last_timestamp) in &self.last_modified_shader_times {
            match *current_modified_shader_times.get(name).unwrap() {
                // If the file was set to 0, we were unable to find it
                // Ex: File was moved or not available, print an error just once if we previously saw it
                0 => {
                    if 0 != *last_timestamp {
                        eprintln!("Unable to access shader file: {}", self.graph.pipelines.get(name.as_str()).unwrap().borrow().info.shader.path);
                    }
                }
                modified_timestamp => {
                    if modified_timestamp != *last_timestamp {
                        unsafe {
                        self.graph.rebuild_pipeline(&name);
                        }
                    }
                }
            }
        }

        self.last_modified_shader_times = current_modified_shader_times;
    }

    pub unsafe fn acquire_swapchain(&mut self) {
        let (present_index, _) = self.swapchain.loader.acquire_next_image(
                self.swapchain.vk,
                std::u64::MAX,
                self.frames[self.frame_index].present_complete_semaphore, // Semaphore to signal
                vk::Fence::null(),
            )
            .unwrap();

        self.present_index = present_index;
    }

    pub fn record_initial_image_load(&self, input_image_buffer: &Buffer) {
        let frame = &self.frames[self.frame_index];
        let device = &self.vk_core.device;

        let buffer_regions = vk::BufferImageCopy {
            buffer_offset: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                layer_count: 1,
                ..Default::default()
            },
            image_extent: vk::Extent3D {
                width: self.info.width as u32,
                height: self.info.height as u32,
                depth: 1
            },
            ..Default::default()
        };

        let input_image = &self.graph.get_input_image();

        /* The goal here is to copy the input file from a vulkan buffer to an srgb image
         * "input_srgb_image" and then to a linear rgb "input_image" so we have the correct
         * gamma */

        // 1. Transition the two input images so they are ready to be transfer destinations
        command::transition_image_layout(&device, frame.cmd_buffer, input_image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        command::transition_image_layout(&device, frame.cmd_buffer, self.input_srgb_image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

        // 2. Copy the buffer to the srgb image and then make it ready to transfer out
        unsafe {
        device.cmd_copy_buffer_to_image(frame.cmd_buffer, input_image_buffer.vk, self.input_srgb_image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[buffer_regions]);
        }
        command::transition_image_layout(&device, frame.cmd_buffer, self.input_srgb_image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

        // 3. Copy the srgb image to the linear input image and get it ready for general compute
        //    Note: If a regular image_copy is used here, we will not get the desired gamma
        //    correction from the format change
        command::blit_copy(&device, frame.cmd_buffer, &command::BlitCopy {
            src_image: self.input_srgb_image.vk, src_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_image: input_image.vk,      dst_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            width: self.info.width,
            height: self.info.height 
        });
        command::transition_image_layout(&device, frame.cmd_buffer, input_image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL);
    }

    pub fn record_pipeline_image_transitions(&self) {
        let graph_frame = &self.graph.frames[self.frame_index];
        let frame = &self.frames[self.frame_index];
        let device = &self.vk_core.device;

        // Transition all intermediate compute images to general
        for (name, image) in &graph_frame.images {
            if name != SWAPCHAIN_OUTPUT && name != FILE_INPUT {
                command::transition_image_layout(&device, frame.cmd_buffer, image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
            }
        }
    }

    pub fn wait_for_frame_fence(&self) {
        let frame = &self.frames[self.frame_index];
        let device = &self.vk_core.device;

        unsafe {
        device
            .wait_for_fences(&[frame.fence], true, std::u64::MAX)
            .expect("Wait for fence failed.");
        }
    }

    pub fn begin_record(&self) {
        let frame = &self.frames[self.frame_index];
        let device = &self.vk_core.device;

        unsafe {
        device.reset_fences(&[frame.fence])
               .expect("Reset fences failed.");

        device.reset_command_buffer(
            frame.cmd_buffer,
            vk::CommandBufferResetFlags::RELEASE_RESOURCES,
        ).expect("Reset command buffer failed.");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();

        device.begin_command_buffer(frame.cmd_buffer, &command_buffer_begin_info)
            .expect("Begin commandbuffer");
        }
    }

    pub fn record(&mut self) {
        let graph_frame = &self.graph.frames[self.frame_index];
        let frame = &mut self.frames[self.frame_index];
        let device = &self.vk_core.device;
        let swapchain_image = self.swapchain.images[self.present_index as usize];

        unsafe {
        device.cmd_reset_query_pool(frame.cmd_buffer,
                                    frame.timer.query_pool,
                                    0, // first-query-idx
                                    frame.timer.query_pool_size);

        command::transition_image_layout(&device, frame.cmd_buffer, graph_frame.images.get(SWAPCHAIN_OUTPUT).unwrap().vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);

        command::execute_pipeline_graph(&device, frame, graph_frame, &self.graph);

        command::transition_image_layout(&device, frame.cmd_buffer, graph_frame.images.get(SWAPCHAIN_OUTPUT).unwrap().vk, vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        command::transition_image_layout(&device, frame.cmd_buffer, swapchain_image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

        /* TODO?: Currently, we are using blit_image because it will do the format
         * conversion for us. However, another alternative is to do copy_image
         * after specifying th final compute shader destination image as the same
         * format as the swapchain format. Maybe worth measuring perf difference later */
        command::blit_copy(device, frame.cmd_buffer, &command::BlitCopy {
            width: self.info.width,
            height: self.info.height,
            src_image: graph_frame.images.get(SWAPCHAIN_OUTPUT).unwrap().vk,
            dst_image: swapchain_image,
            src_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL
        });

        command::transition_image_layout(&device, frame.cmd_buffer, swapchain_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);

        }
    }

    pub fn end_record(&self) {
        unsafe {
        self.vk_core.device.end_command_buffer(self.frames[self.frame_index].cmd_buffer).unwrap();
        }
    }

    pub fn submit(&mut self) {
        let frame = &self.frames[self.frame_index];

        let present_complete_semaphore = &[frame.present_complete_semaphore];
        let cmd_buffers = &[frame.cmd_buffer];
        let signal_semaphores = &[frame.render_complete_semaphore];

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(present_complete_semaphore)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COMPUTE_SHADER])
            .command_buffers(cmd_buffers)
            .signal_semaphores(signal_semaphores);

        unsafe {
        self.vk_core.device.queue_submit(
            self.vk_core.queue,
            &[submit_info.build()],
            frame.fence,
        ).expect("queue submit failed.");
        }

        let wait_semaphores = [frame.render_complete_semaphore];
        let swapchains = [self.swapchain.vk];
        let image_indices = [self.present_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
        self.swapchain.loader
            .queue_present(self.vk_core.queue, &present_info)
            .unwrap();
        }

        self.frame_index = (self.frame_index+1)%self.info.num_frames;
    }

    pub fn last_frame_gpu_times(&mut self) -> String {
        self.frames[self.frame_index].timer.get_elapsed_ms()
    }


    pub fn new(info: RenderInfo, event_loop: &EventLoop<()>) -> Render {
        let window = Self::create_window(&event_loop, info.width, info.height);

        let pipeline_config = Self::load_config(&info.config_path).unwrap();

        unsafe {
        let vk_core = VkCore::new(&window);

        let graph = Self::create_graph(&vk_core, &info, &pipeline_config).unwrap();

        let frames : Vec<Frame> = (0..info.num_frames).map(|_|{
            Frame::new(&vk_core, pipeline_config.len() as u32)
        }).collect();

        let input_srgb_image = vkutils::create_image(&vk_core,
                                                     "input-image-srgb".to_string(),
                                                     vk::Format::R8G8B8A8_SRGB,
                                                     info.width, info.height);

        let last_modified_shader_times: HashMap<String, u64> = utils::get_modified_times(&graph.pipelines);
        let last_modified_config_time: u64 = if let Some(node_config) = info.config_path.as_ref() { utils::get_modified_time(node_config) } else { 0 };

        let swapchain = SwapChain::new(&vk_core, info.width, info.height);


        Render {
            frames: frames,
            graph: graph,
            info: info,
            input_srgb_image: input_srgb_image,
            last_modified_config_time: last_modified_config_time,
            last_modified_shader_times: last_modified_shader_times,
            present_index: 0,
            frame_index: 0,
            vk_core: vk_core,
            swapchain: swapchain,
            _window: window
        }

        }
    }
}

impl Drop for Render {
    fn drop(&mut self) {
        unsafe {
            self.vk_core.device.device_wait_idle().unwrap();
        }
    }
}
