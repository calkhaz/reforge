use ash::vk;
use gpu_allocator as gpu_alloc;

use crate::config::config::Config;
use crate::config::config::parse_file as config_file_parse;
use crate::config::config::single_shader_parse as config_single_shader_parse;
use crate::utils;
use crate::vulkan::command;
use crate::vulkan::core::VkCore;
use crate::vulkan::frame::Frame;
use crate::vulkan::pipeline_graph::FILE_INPUT;
use crate::vulkan::pipeline_graph::PipelineGraph;
use crate::vulkan::pipeline_graph::PipelineGraphInfo;
use crate::vulkan::pipeline_graph::FINAL_OUTPUT;
use crate::vulkan::swapchain::SwapChain;
use crate::vulkan::vkutils;
use crate::vulkan::vkutils::Buffer;
use crate::vulkan::vkutils::Image;
use crate::warnln;

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
    pub shader_path: String,
    pub format: vk::Format,
    pub swapchain: bool,
    pub has_input_image: bool,
    pub shader_file_path: Option<String>
}

pub struct Render {
    frames: Vec<Frame>,
    graph: PipelineGraph,
    info: RenderInfo,
    // Used to bring buffer -> srgba8 -> X or X -> srgba8 -> buffer
    staging_srgb_image: Image,
    pub staging_buffer: Buffer,
    last_modified_config_time: u64,
    last_modified_shader_times: HashMap<String, u64>,
    present_index: u32,
    pub frame_index: usize,
    swapchain: Option<SwapChain>,
    window: Option<winit::window::Window>,
    pub vk_core: VkCore,
    pub swapchain_rebuilt_required: bool
}

impl Render {
    pub fn staging_buffer_ptr(&mut self) -> *mut u8 {
        self.staging_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8
    }

    fn get_swapchain(&self) -> &SwapChain {
        &self.swapchain.as_ref().expect("No swapchain created")
    }

    fn create_window(event_loop: &EventLoop<()>, width: u32, height:u32) -> winit::window::Window {
        WindowBuilder::new()
            .with_title("Reforge")
            .with_inner_size(winit::dpi::PhysicalSize::new(
                f64::from(width),
                f64::from(height),
            ))
            .with_resizable(true)
            .build(&event_loop)
            .unwrap()
    }

    unsafe fn create_graph(vk_core: &VkCore, info: &RenderInfo, pipeline_config: &Config) -> Option<PipelineGraph> {
        let pipeline_infos = vkutils::synthesize_config(Rc::clone(&vk_core.device), &pipeline_config, &info.shader_path)?;

        let graph_info = PipelineGraphInfo {
            pipeline_infos: pipeline_infos,
            format: info.format,
            width: info.width,
            height: info.height,
            num_frames: info.num_frames
        };

        let mut graph = PipelineGraph::new(&vk_core, graph_info)?;

        for (pipeline_name, instance) in &pipeline_config.pipeline_instances {
            Self::initialize_ubos(&mut graph, pipeline_name, &instance.parameters);
        }

        Some(graph)
    }

    fn create_config(info: &RenderInfo) -> Option<Config>{
        match info.config_path.as_ref() {
            // Read from provided configuration file
            Some(path) => { 
                let contents = utils::load_file_contents(&path);
                if contents.is_none() { warnln!("Empty configuration file"); return None }
                config_file_parse(contents.unwrap(), info.has_input_image, &info.shader_path)
            }
            // No configuration file path provided
            None => {
                match info.shader_file_path.as_ref() {
                    // Create a configuration for just a single provided shader file
                    Some(path) => { Some(config_single_shader_parse(path.clone(), info.has_input_image)) }

                    // Use the default passthrough configuration
                    None => { config_file_parse("input -> passthrough -> output".to_string(), true, &info.shader_path) }
                }
            }
        }
    }

    fn recreate_graph(&mut self) -> Option<()> {
        let pipeline_config = Self::create_config(&self.info)?;

        unsafe {
        self.vk_core.device.device_wait_idle().unwrap();

        let num_pipelines = pipeline_config.graph_pipelines.len() as u32;
        let graph = Self::create_graph(&self.vk_core, &self.info, &pipeline_config)?;

        self.graph = graph;
        self.frames.iter_mut().for_each(|f| f.rebuild_timer(num_pipelines));
        }
        self.frame_index = 0;

        Some(())
    }

    fn config_changed(&mut self) -> bool {
        if self.info.config_path.is_none() {
            return false;
        }
        let config_path = self.info.config_path.as_ref().unwrap();

        let current_modified_config_time = utils::get_modified_time(&config_path);

        match current_modified_config_time {
            0 => {
                if 0 != self.last_modified_config_time {
                    warnln!("Unable to access config file: {}", config_path);
                }
            },
            modified_timestamp => {
                if modified_timestamp == self.last_modified_config_time {
                    return false;
                }

                self.last_modified_config_time = current_modified_config_time;
                self.last_modified_shader_times = utils::get_modified_times(&self.graph.pipelines);

                return true;
            }
        };

        false
    }

    pub fn initialize_ubos(graph: &mut PipelineGraph, pipeline_name: &String, parameters: &HashMap<String, String>) {

        let write_to_buffer = |value_str: &String, ptr: *mut u8, block_type: spirv_reflect::types::ReflectTypeFlags | {
            match block_type {
                spirv_reflect::types::ReflectTypeFlags::FLOAT => {
                    let value = value_str.parse::<f32>().unwrap_or_else(|e| { warnln!("Failed to convert: {}", e); 0.0 });
                    unsafe { std::ptr::copy_nonoverlapping(&value, ptr as *mut f32, 1); }
                },
                spirv_reflect::types::ReflectTypeFlags::INT => {
                    let value = value_str.parse::<i32>().unwrap_or_else(|e| { warnln!("Failed to convert: {}", e); 0 });
                    unsafe { std::ptr::copy_nonoverlapping(&value, ptr as *mut i32, 1); }
                },
                spirv_reflect::types::ReflectTypeFlags::BOOL => {
                    let value = value_str.parse::<bool>().unwrap_or_else(|e| { warnln!("Failed to convert: {}", e); false });
                    unsafe { std::ptr::copy_nonoverlapping(&value, ptr as *mut bool, 1); }
                },
                _ => {}
            };
        };

        for frame in &mut graph.frames {
            if let Some(ubo) = frame.ubos.get_mut(pipeline_name) {
                for (buffer_member_name, buffer_block) in ubo {
                    if buffer_member_name == "_rf_time" {
                        continue;
                    }

                    unsafe {
                    let ptr = buffer_block.buffer.mapped_data.offset(buffer_block.offset as isize);

                    if let Some(param_value) = parameters.get(buffer_member_name) {
                        write_to_buffer(param_value, ptr, buffer_block.block_type);
                    }
                    else {
                        let value = 0;
                        std::ptr::copy_nonoverlapping(&value, ptr as *mut u8, buffer_block.size as usize);
                    }
                    }


                }
            }
        }
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

    fn reload_changed_pipelines(&mut self) {
        let current_modified_shader_times: HashMap<String, u64> = utils::get_modified_times(&self.graph.pipelines);

        for (name, last_timestamp) in &self.last_modified_shader_times {
            match *current_modified_shader_times.get(name).unwrap() {
                // If the file was set to 0, we were unable to find it
                // Ex: File was moved or not available, print an error just once if we previously saw it
                0 => {
                    if 0 != *last_timestamp {
                        let pipeline = self.graph.pipelines.get(name.as_str()).unwrap().borrow();
                        warnln!("Unable to access shader file: {}", pipeline.info.shader.borrow().path.as_ref().unwrap());
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
        let swapchain = self.get_swapchain();
        let (present_index, _) = swapchain.loader.acquire_next_image(
                swapchain.vk,
                std::u64::MAX,
                self.frames[self.frame_index].present_complete_semaphore, // Semaphore to signal
                vk::Fence::null(),
            )
            .unwrap();

        self.present_index = present_index;
    }

    pub fn record_initial_image_load(&self) {
        let frame = &self.frames[self.frame_index];
        let device = &self.vk_core.device;
        let graph_frame = &self.graph.frames[self.frame_index];

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

        let input_image = &graph_frame.get_input_image();

        /* The goal here is to copy the input file from a vulkan buffer to an srgb image
         * "staging_srgb_image" and then to a linear rgb "input_image" so we have the correct
         * gamma */

        // 1. Transition the two input images so they are ready to be transfer destinations
        command::transition_image_layout(&device, frame.cmd_buffer, input_image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        command::transition_image_layout(&device, frame.cmd_buffer, self.staging_srgb_image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

        // 2. Copy the buffer to the srgb image and then make it ready to transfer out
        unsafe {
        device.cmd_copy_buffer_to_image(frame.cmd_buffer, self.staging_buffer.vk, self.staging_srgb_image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[buffer_regions]);
        }
        command::transition_image_layout(&device, frame.cmd_buffer, self.staging_srgb_image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

        // 3. Copy the srgb image to the linear input image and get it ready for general compute
        //    Note: If a regular image_copy is used here, we will not get the desired gamma
        //    correction from the format change
        command::blit_copy(&device, frame.cmd_buffer, &command::BlitCopy {
            src_width:  self.info.width,
            src_height: self.info.height,
            dst_width:  self.info.width,
            dst_height: self.info.height,
            src_image:  self.staging_srgb_image.vk, src_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_image:  input_image.vk, dst_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ..Default::default()
        });
        command::transition_image_layout(&device, frame.cmd_buffer, input_image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::GENERAL);
    }

    pub fn record_pipeline_image_transitions(&self) {
        let graph_frame = &self.graph.frames[self.frame_index];
        let frame = &self.frames[self.frame_index];
        let device = &self.vk_core.device;

        // Transition all intermediate compute images to general
        for (name, image) in &graph_frame.images {
            if name != FINAL_OUTPUT && name != FILE_INPUT {
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
        let swapchain_image = if self.swapchain.is_some() { Some(self.get_swapchain().images[self.present_index as usize]) } else { None };
        let frame = &mut self.frames[self.frame_index];
        let device = &self.vk_core.device;

        unsafe {
        device.cmd_reset_query_pool(frame.cmd_buffer,
                                    frame.timer.query_pool,
                                    0, // first-query-idx
                                    frame.timer.query_pool_size);

        if self.graph.is_compute() {
            command::transition_image_layout(&device, frame.cmd_buffer, graph_frame.get_output_image(), vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
        }

        command::execute_pipeline_graph(&device, frame, graph_frame, &self.graph);

        if self.graph.is_compute() {
            command::transition_image_layout(&device, frame.cmd_buffer, graph_frame.get_output_image(), vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        }

        if self.swapchain.is_some() {
            command::transition_image_layout(&device, frame.cmd_buffer, swapchain_image.unwrap(), vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

            /* TODO?: Currently, we are using blit_image because it will do the format
             * conversion for us. However, another alternative is to do copy_image
             * after specifying th final compute shader destination image as the same
             * format as the swapchain format. Maybe worth measuring perf difference later */
            command::blit_copy(device, frame.cmd_buffer, &command::BlitCopy {
                src_width: if self.info.has_input_image { self.info.width } else { self.swapchain.as_ref().unwrap().width },
                src_height: if self.info.has_input_image { self.info.height } else { self.swapchain.as_ref().unwrap().height },
                dst_width: self.swapchain.as_ref().unwrap().width,
                dst_height: self.swapchain.as_ref().unwrap().height,
                src_image: graph_frame.get_output_image(),
                dst_image: swapchain_image.unwrap(),
                src_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                center: true
            });

            command::transition_image_layout(&device, frame.cmd_buffer, swapchain_image.unwrap(), vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
        }

        }
    }

    pub fn write_output_to_buffer(&self) {
        let graph_frame = &self.graph.frames[self.frame_index];
        let frame = &self.frames[self.frame_index];
        let device = &self.vk_core.device;

        command::transition_image_layout(&device, frame.cmd_buffer, self.staging_srgb_image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

        command::blit_copy(device, frame.cmd_buffer, &command::BlitCopy {
            src_width:  self.info.width,
            src_height: self.info.height,
            dst_width:  self.info.width,
            dst_height: self.info.height,
            src_image: graph_frame.get_output_image(),
            dst_image: self.staging_srgb_image.vk,
            src_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ..Default::default()
        });

        command::transition_image_layout(&device, frame.cmd_buffer, self.staging_srgb_image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

        command::copy_image_to_buffer(device, frame.cmd_buffer, &command::ImageToBuffer {
            width: self.info.width,
            height: self.info.height,
            src_image: &self.staging_srgb_image,
            dst_buffer: self.staging_buffer.vk
        })
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

        let mut submit_info = vk::SubmitInfo::builder()
            .command_buffers(cmd_buffers);

        // Semaphores only needed if we use a swapchain
        if self.swapchain.is_some() {
            submit_info = submit_info
                .wait_dst_stage_mask(&[vk::PipelineStageFlags::COMPUTE_SHADER])
                .wait_semaphores(present_complete_semaphore)
                .signal_semaphores(signal_semaphores);
        }

        unsafe {
        self.vk_core.device.queue_submit(
            self.vk_core.queue,
            &[submit_info.build()],
            frame.fence,
        ).expect("queue submit failed.");
        }

        if self.swapchain.is_some() {
            let wait_semaphores = [frame.render_complete_semaphore];
            let swapchain = self.get_swapchain();
            let swapchains = [swapchain.vk];
            let image_indices = [self.present_index];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            unsafe {

            match swapchain.loader.queue_present(self.vk_core.queue, &present_info) {
                Ok(suboptimal) => {
                    if suboptimal {
                        self.swapchain_rebuilt_required = true;
                    }
                },
                Err(err) => {
                    if err == vk::Result::ERROR_OUT_OF_DATE_KHR {
                        self.swapchain_rebuilt_required = true;
                    }
                }
            }
            }
        }

        self.frame_index = (self.frame_index+1)%self.info.num_frames;
    }

    pub fn trigger_reloads(&mut self) -> bool {
        let mut full_reload_performed = false;

        // If the window has changed, we need to reload the swapchain
        if self.swapchain_rebuilt_required {
            self.rebuild_swapchain();
            full_reload_performed = self.recreate_graph().is_some();
            self.swapchain_rebuilt_required = false;
        }

        // If our configuration has changed, live reload it
        if self.config_changed() {
            full_reload_performed = self.recreate_graph().is_some();
        }

        // If any of our shaders have changed, live reload them
        if full_reload_performed {
            self.last_modified_shader_times.clear();
        }
        self.reload_changed_pipelines();

        full_reload_performed
    }

    pub fn last_frame_gpu_times(&mut self) -> String {
        self.frames[self.frame_index].timer.get_elapsed_ms()
    }

    fn rebuild_swapchain(&mut self) {
        let window_size = self.window.as_ref().unwrap().inner_size();
        unsafe {
        self.vk_core.device.device_wait_idle().unwrap();
        if !self.info.has_input_image {
            self.info.width = window_size.width;
            self.info.height = window_size.height;
        }
        self.swapchain.as_mut().unwrap().rebuild(&self.vk_core, window_size.width, window_size.height);
        }
    }

    pub fn new(info: RenderInfo, event_loop: &Option<EventLoop<()>>) -> Render {
        let window = if info.swapchain { Some(Self::create_window(&event_loop.as_ref().unwrap(), info.width, info.height)) } else { None };

        let pipeline_config = Self::create_config(&info).unwrap();

        unsafe {
        let vk_core = VkCore::new(&window);

        let graph = Self::create_graph(&vk_core, &info, &pipeline_config).unwrap();

        let frames : Vec<Frame> = (0..info.num_frames).map(|_|{
            Frame::new(&vk_core, pipeline_config.graph_pipelines.len() as u32)
        }).collect();

        // We use rgba8 as the input file format
        let buffer_size = (info.width as vk::DeviceSize)*(info.height as vk::DeviceSize)*4;

        // This staging buffer will be used to transfer the original input file into an mimage
        let staging_buffer = vkutils::create_buffer(&vk_core,
                                                        "input-image-staging-buffer".to_string(),
                                                        buffer_size,
                                                        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                                                        gpu_alloc::MemoryLocation::CpuToGpu);

        let staging_srgb_image = vkutils::create_image(&vk_core,
                                                       "input-image-srgb".to_string(),
                                                       vk::Format::R8G8B8A8_SRGB,
                                                       info.width, info.height);

        let last_modified_shader_times: HashMap<String, u64> = utils::get_modified_times(&graph.pipelines);
        let last_modified_config_time: u64 = if let Some(pipeline_config) = info.config_path.as_ref() { utils::get_modified_time(pipeline_config) } else { 0 };

        let swapchain = if info.swapchain { Some(SwapChain::new(&vk_core, info.width, info.height)) } else { None };

        Render {
            frames: frames,
            graph: graph,
            info: info,
            staging_srgb_image: staging_srgb_image,
            staging_buffer: staging_buffer,
            last_modified_config_time: last_modified_config_time,
            last_modified_shader_times: last_modified_shader_times,
            present_index: 0,
            frame_index: 0,
            vk_core: vk_core,
            swapchain: swapchain,
            window: window,
            swapchain_rebuilt_required: false
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
