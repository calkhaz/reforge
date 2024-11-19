use anyhow::{anyhow, Context, Result};
use ash::vk;
use gpu_allocator as gpu_alloc;

use crate::err;
use crate::ui;
use crate::utils;
use crate::config::Config;
use crate::vulkan::command;
use crate::vulkan::core::VkCore;
use crate::vulkan::frame::Frame;
use crate::vulkan::pipeline::Pipeline;
use crate::vulkan::pipeline_graph::BufferBlock;
use crate::vulkan::pipeline_graph::FILE_INPUT;
use crate::vulkan::pipeline_graph::PipelineGraph;
use crate::vulkan::pipeline_graph::PipelineGraphInfo;
use crate::vulkan::pipeline_graph::FINAL_OUTPUT;
use crate::vulkan::swapchain::SwapChain;
use crate::vulkan::vkutils;
use crate::vulkan::vkutils::Buffer;
use crate::vulkan::vkutils::Image;
use crate::vulkan::shader::DescBlockType;
use crate::vulkan::render_pass;
use tracing::warn;

use std::collections::HashMap;
use std::default::Default;
use std::rc::Rc;
use winit::window::Window;

#[derive(Clone, Debug)]
pub enum ParamData {
    Boolean(bool),
    Float(f32),
    Integer(i32),
    FloatArray(Vec<f32>),
    IntegerArray(Vec<i32>)
}

/// Orthographic projection matrix for Vulkan
/// From: https://github.com/fu5ha/ultraviolet
#[inline]
pub fn orthographic_vk(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> [f32; 16] {
    let rml = right - left;
    let rpl = right + left;
    let tmb = top - bottom;
    let tpb = top + bottom;
    let fmn = far - near;

    #[rustfmt::skip]
    let res = [
        2.0 / rml, 0.0, 0.0, 0.0,
        0.0, -2.0 / tmb, 0.0, 0.0,
        0.0, 0.0, -1.0 / fmn, 0.0,
        -(rpl / rml), -(tpb / tmb), -(near / fmn), 1.0
    ];

    res
}

unsafe fn any_as_u8_slice<T: Sized>(any: &T) -> &[u8] {
    let ptr = (any as *const T) as *const u8;
    std::slice::from_raw_parts(ptr, std::mem::size_of::<T>())
}



impl ParamData {
    fn primitive<T: num_traits::cast::NumCast>(&self) -> Result<T> {
        let prim = match self {
            ParamData::Float(v)   => { num_traits::cast::<f32, T>(*v) },
            ParamData::Integer(v) => { num_traits::cast::<i32, T>(*v) },
            ParamData::Boolean(v) => { let tmp = *v as u32;
                                       num_traits::cast::<u32, T>(tmp) },
            _ => None
        };

        prim.context(format!("Unable to parse primitive {:?}", self))
    }

    fn write_to_buffer<T: num_traits::cast::NumCast>(&self, buffer: *mut u8) -> Result<()> {
        let val = self.primitive::<T>()?;
        unsafe { std::ptr::copy_nonoverlapping(&val, buffer as *mut T, 1); }
        Ok(())
    }

    fn primitive_vec<T: num_traits::cast::NumCast>(&self) -> Result<Vec<T>> {
        let prim_vec = match self {
            ParamData::FloatArray(v)   => {
                Some(v.iter().filter_map(|val|  { num_traits::cast::<f32, T>(*val) }).collect())
            },
            ParamData::IntegerArray(v) => {
                Some(v.iter().filter_map(|val|  { num_traits::cast::<i32, T>(*val) }).collect())
            },
            _ => None
        };

        prim_vec.context(format!("Unable to parse vec primitive {:?}", self))
    }

    fn write_vec_to_buffer<T: num_traits::cast::NumCast>(&self, buffer: *mut u8, block: &BufferBlock) -> Result<()> {
        let vec = self.primitive_vec::<T>()?;
        let mut offset = 0;

        if vec.len() > block.ubo.array_len as usize {
            return err!("Vector exceeds block size");
        }

        for v in vec {
            let offset_buffer = unsafe { buffer.offset(offset as isize) };
            unsafe { std::ptr::copy_nonoverlapping(&v, offset_buffer as *mut T, 1); }
            offset += block.ubo.array_stride;
        }

        Ok(())
    }
}

pub struct RenderInfo {
    pub config: Config,
    pub width: u32,
    pub height: u32,
    pub num_frames: usize,
    pub format: vk::Format,
    pub has_input_image: bool,
}

struct UiResources {
    render_pass: vk::RenderPass,
    pipeline: Pipeline,
    _descriptor_pool: vk::DescriptorPool,
    font_image: Option<Image>,
    font_image_staging_buffer: Option<Buffer>,
    descriptor_set: vk::DescriptorSet,
}

struct UiPerSwapchain {
    pub framebuffer: vk::Framebuffer,
    pub vertex_buffer: vkutils::Buffer,
    pub index_buffer: vkutils::Buffer,
}

struct PerSwapChainRes {
    ui: UiPerSwapchain
}

impl PerSwapChainRes {
    pub fn new(core: &VkCore, ui_render_pass: vk::RenderPass, sc_image_view: vk::ImageView, width: u32, height: u32) -> Result<PerSwapChainRes> {
        unsafe {
        let ui : UiPerSwapchain = {
            //let image = vkutils::create_image(core, "ui-image".to_string(), format, width, height);
            let framebuffer = render_pass::build_framebuffer(core, sc_image_view, ui_render_pass, width, height)?;
            let vertex_buffer = vkutils::create_buffer(core, "ui-vertex".to_string(), 32*1024*1024, vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC, gpu_allocator::MemoryLocation::GpuToCpu);
            let index_buffer  = vkutils::create_buffer(core, "ui-index".to_string(), 32*1024*1024, vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC, gpu_allocator::MemoryLocation::GpuToCpu);

            UiPerSwapchain{ framebuffer, vertex_buffer, index_buffer}
        };

        Ok(PerSwapChainRes { ui })
        }
    }
}

pub struct Render {
    frames: Vec<Frame>,
    frame_outdated: Vec<bool>,
    graph: PipelineGraph,
    info: RenderInfo,
    // Used to bring buffer -> srgba8 -> X or X -> srgba8 -> buffer
    staging_srgb_image: Image,
    pub staging_buffer: Buffer,
    last_modified_shader_times: HashMap<String, u64>,
    present_index: u32,
    pub frame_index: usize,
    swapchain: Option<SwapChain>,
    pub swapchain_rebuilt_required: bool,
    pub pipeline_buffer_data: HashMap<String, HashMap<String, ParamData>>,
    reload_config: Option<Config>,
    pub window_width: u32,
    pub window_height: u32,
    pub ui: Option<ui::Ui>,
    ui_res: Option<UiResources>,
    per_sc_res: Option<Vec<PerSwapChainRes>>,
    pub vk_core: VkCore,
}

impl UiResources {
    pub fn new(device: &Rc<ash::Device>, format: vk::Format) -> Result<UiResources> {
        unsafe {
        // Going from vulkan blit as dst to presentation
        let render_pass = render_pass::build_render_pass(&device, format, vk::AttachmentLoadOp::LOAD, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR)?;

        let pipeline = Pipeline::new_ui_gfx(device.clone(), render_pass)?;

        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1
        };

        let pool_vec = &[pool_size];

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(pool_vec)
            .max_sets(1u32);

        let descriptor_pool = device
            .create_descriptor_pool(&descriptor_pool_info, None)?;

        let desc_layouts = &[pipeline.layout.descriptor_layout];

        let desc_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(desc_layouts);

        let descriptor_set = device
            .allocate_descriptor_sets(&desc_alloc_info)
            .unwrap()[0];

        Ok(UiResources { render_pass, pipeline, _descriptor_pool: descriptor_pool, font_image: None, font_image_staging_buffer: None, descriptor_set })
        }
    }
}

impl Render {
    pub fn has_swapchain(&self) -> bool {
        self.swapchain.is_some()
    }

    pub fn outdate_frames(&mut self) {
        for f in &mut self.frame_outdated {
            *f = true;
        }
    }

    pub fn staging_buffer_ptr(&mut self) -> *mut u8 {
        self.staging_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8
    }

    pub fn update_config(&mut self, config: Config) {
        self.reload_config = Some(config);
    }

    fn get_swapchain(&self) -> &SwapChain {
        &self.swapchain.as_ref().expect("No swapchain created")
    }

    unsafe fn create_graph(vk_core: &VkCore, info: &RenderInfo) -> Result<PipelineGraph> {
        let pipeline_infos = vkutils::synthesize_config(Rc::clone(&vk_core.device), &info.config)?;

        let graph_info = PipelineGraphInfo {
            pipeline_infos,
            format: info.format,
            width: info.width,
            height: info.height,
            num_frames: info.num_frames
        };

        PipelineGraph::new(&vk_core, graph_info)
    }

    fn recreate_graph(&mut self) -> Result<()> {
        unsafe {
        self.vk_core.device.device_wait_idle()?;

        let num_pipelines = self.info.config.graph_pipelines.len() as u32;
        let graph = Self::create_graph(&self.vk_core, &self.info)?;

        self.graph = graph;
        self.frames.iter_mut().for_each(|f| f.rebuild_timer(num_pipelines));
        }
        self.frame_index = 0;

        Ok(())
    }

    pub fn write_to_outdated_ubos(&mut self) -> Result<()> {
        let outdated = self.frame_outdated[self.frame_index];
    
        if !outdated {
            return Ok(())
        }
    
        let ubos = &mut self.graph.frames[self.frame_index].ubos;
    
        let write_to_buffer = |val: &ParamData, ptr: *mut u8, block: &BufferBlock | -> Result<()> {
            let t = block.ubo.block_type;
    
            // Array primitives
            if      t == DescBlockType::FLOAT | DescBlockType::ARRAY  { val.write_vec_to_buffer::<f32>(ptr, block)?; }
            else if t == DescBlockType::INT   | DescBlockType::ARRAY  { val.write_vec_to_buffer::<i32>(ptr, block)?; }
            // Single primitives
            else if t == DescBlockType::FLOAT { val.write_to_buffer::<f32>(ptr)?; }
            else if t == DescBlockType::INT   { val.write_to_buffer::<i32>(ptr)?; }
            else if t == DescBlockType::BOOL  { val.write_to_buffer::<u32>(ptr)?; }
    
            Ok(())
        };

        // Descripting debugging
        // println!("ubos: {:#?}", ubos);
        // println!("pipeline_buffer_data: {:#?}", self.pipeline_buffer_data);
    
        // For every pipeline, pair parameter and bufferblock hashmaps by pipeline name
        let matched_pipelines: Vec<(&HashMap<String, ParamData>, &HashMap<String, BufferBlock>)> =
            ubos.iter().filter_map(|(name, buffer_map)| {
                match self.pipeline_buffer_data.get(name) {
                    Some(param_map) => Some((param_map, buffer_map)),
                    None => None
                }}).collect();
    
        // For every pipeline (1st vec), we'll want vector (2nd vec)
        // for each parameter containing a tuple of the param name, param data, and buffer block
        let matched_params: Vec<Vec<(&String, &ParamData, &BufferBlock)>> =
            matched_pipelines.iter().filter_map(|(param_map, buffer_map)|{
    
                // Couple parameter and buffer block by name and flatten into vector
                let combined: Vec<(&String, &ParamData, &BufferBlock)> =
                    param_map.iter().filter_map(|(name, param)| {
                        match buffer_map.get(name) {
                            Some(buffer) => Some((name, param, buffer)),
                            None => None
                    }}).collect();
    
                // Don't add empty vectors
                if combined.is_empty() { None } else { Some(combined) }
    
            }).collect();
    
        for (name, param, buffer_block) in matched_params.iter().flatten() {
            let ptr = unsafe { buffer_block.buffer.mapped_data.offset(buffer_block.ubo.offset as isize) };
    
            write_to_buffer(param, ptr, &buffer_block).context(format!("Failed to write param to buffer {}", name))?;
        }
    
        self.frame_outdated[self.frame_index] = false;

        Ok(())
    }

    pub fn update_ubos(&mut self, time: f32) -> Result<()> {
        self.write_to_outdated_ubos()?;
    
        self.graph.frames[self.frame_index].ubos.iter_mut().for_each(|(_pipeline_name, buffer_block_map)| {
            buffer_block_map.iter_mut().for_each(|(buffer_member_name, buffer_block)| {
                if buffer_member_name.ends_with("_rf_time") {
                    unsafe {
                    let ptr = buffer_block.buffer.mapped_data.offset(buffer_block.ubo.offset as isize) as *mut f32;
                    std::ptr::copy_nonoverlapping(&time, ptr, 1)
                    }
                }
            });
        });

        Ok(())
    }

    fn reload_changed_pipelines(&mut self) {
        let current_modified_shader_times: HashMap<String, u64> = utils::get_modified_times(&self.graph.pipelines);

        for (name, last_timestamp) in &self.last_modified_shader_times {
            match *current_modified_shader_times.get(name).unwrap() {
                // If the file was set to 0, we were unable to find it
                // Ex: File was moved or not available, print an error just once if we previously saw it
                0 => {
                    if 0 != *last_timestamp {
                        let pipeline = self.graph.pipelines.get(name).unwrap().borrow();
                        warn!("Unable to access shader file: {}", pipeline.info.shader.borrow().path.as_ref().unwrap());
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

    pub fn acquire_swapchain(&mut self) -> Result<()> {
        let swapchain = self.get_swapchain();
        unsafe {
        let (present_index, _) = swapchain.loader.acquire_next_image(
                swapchain.vk,
                std::u64::MAX,
                self.frames[self.frame_index].present_complete_semaphore, // Semaphore to signal
                vk::Fence::null(),
            )?;

        self.present_index = present_index;
        }

        Ok(())
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

    fn record_ui_font_image_upload(&mut self, font_image: egui::FontImage) {
        let ui_res = self.ui_res.as_mut().unwrap();

        let data = font_image
            .srgba_pixels(None)
            .flat_map(|c| c.to_array())
            .collect::<Vec<_>>();

        let data = data.as_slice();
        let device = &self.vk_core.device;
        let frame = &self.frames[self.frame_index];

        unsafe {
        let buffer = vkutils::create_buffer(&self.vk_core, "ui-font-staging".to_string(), (data.len()) as u64, vk::BufferUsageFlags::TRANSFER_SRC, gpu_alloc::MemoryLocation::GpuToCpu);
        std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.mapped_data, data.len());
        let image = vkutils::create_image(&self.vk_core, "ui-font".to_string(), vk::Format::R8G8B8A8_SRGB, font_image.width() as u32, font_image.height() as u32);


        let buffer_regions = vk::BufferImageCopy {
            buffer_offset: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                layer_count: 1,
                ..Default::default()
            },
            image_extent: vk::Extent3D {
                width: font_image.width() as u32,
                height: font_image.height() as u32,
                depth: 1
            },
            ..Default::default()
        };

        command::transition_image_layout(&device, frame.cmd_buffer, image.vk, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        device.cmd_copy_buffer_to_image(frame.cmd_buffer, buffer.vk, image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[buffer_regions]);
        command::transition_image_layout(&device, frame.cmd_buffer, image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let desc_info = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image_view: image.view,
            sampler: self.graph.sampler.vk
        };

        let descriptor_write = vk::WriteDescriptorSet {
            dst_set: ui_res.descriptor_set,
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info: [desc_info].as_ptr(),
            ..Default::default()
        };

        device.update_descriptor_sets(&[descriptor_write], &[]);

        ui_res.font_image_staging_buffer = Some(buffer);
        ui_res.font_image = Some(image);
        }
    }

    fn record_ui_drawing(&self, primitives: &Vec<egui::ClippedPrimitive>, pixels_per_point: f32) {
        let frame = &self.frames[self.frame_index];
        let ui_sc = &self.per_sc_res.as_ref().unwrap()[self.present_index as usize].ui;
        let device = &self.vk_core.device;

        let mut index_write_offset  = 0usize;
        let mut vertex_write_offset = 0usize;
        let mut index_draw_offset  = 0u32;
        let mut vertex_draw_offset = 0u32;

        for p in primitives {
            let clip_rect = p.clip_rect;
            match &p.primitive {
                egui::epaint::Primitive::Mesh(m) => {
                    let clip_x = clip_rect.min.x * pixels_per_point;
                    let clip_y = clip_rect.min.y * pixels_per_point;
                    let clip_w = clip_rect.max.x * pixels_per_point - clip_x;
                    let clip_h = clip_rect.max.y * pixels_per_point - clip_y;

                    let scissors = [vk::Rect2D {
                        offset: vk::Offset2D {
                            x: (clip_x as i32).max(0),
                            y: (clip_y as i32).max(0),
                        },
                        extent: vk::Extent2D {
                            width: clip_w as _,
                            height: clip_h as _,
                        },
                    }];

                    unsafe {
                        device.cmd_set_scissor(frame.cmd_buffer, 0, &scissors);
                    }

                    let index_count = m.indices.len() as u32;
                    let vertex_size = std::mem::size_of::<egui::epaint::Vertex>();
                    let index_size = std::mem::size_of::<u32>();

                    unsafe {
                        device.cmd_draw_indexed(
                            frame.cmd_buffer,
                            index_count,
                            1,
                            index_draw_offset,
                            vertex_draw_offset as i32,
                            0,
                        );

                        let vertex_ptr = ui_sc.vertex_buffer.mapped_data.offset(vertex_write_offset as isize);
                        let index_ptr  = ui_sc.index_buffer.mapped_data.offset(index_write_offset as isize);

                        std::ptr::copy_nonoverlapping(m.vertices.as_ptr() as *const u8, vertex_ptr, vertex_size*m.vertices.len());
                        std::ptr::copy_nonoverlapping(m.indices.as_ptr()  as *const u8, index_ptr,  index_size*m.indices.len());
                    };

                    index_draw_offset  += m.indices.len() as u32;
                    vertex_draw_offset += m.vertices.len() as u32;
                    index_write_offset  += index_size*m.indices.len();
                    vertex_write_offset += vertex_size*m.vertices.len();
                }
                egui::epaint::Primitive::Callback(_) => {
                    warn!("Primitive callbacks are not supported")
                }
            }
        }
    }

    fn record_ui(&mut self) {
        let (primitives, texture_deltas, pixels_per_point) = self.ui.as_mut().unwrap().run();

        // Only upload font image once ever
        for (_texture_id, image_delta) in texture_deltas.set {
            match image_delta.image {
                egui::ImageData::Color(_) => { warn!("egui color image delta is unsupported"); }
                egui::ImageData::Font(f) => {
                    if self.ui_res.as_ref().unwrap().font_image.is_none() {
                        self.record_ui_font_image_upload(f);
                    }
                    else {
                        warn!("Egui tried to upload font more than once, but we don't handle that case");
                    }
                }
            };
        }

        let frame = &self.frames[self.frame_index];
        let ui_sc = &self.per_sc_res.as_ref().unwrap()[self.present_index as usize].ui;
        let swapchain = self.swapchain.as_ref().unwrap();
        let device = &self.vk_core.device;
        let ui_res = &self.ui_res.as_ref().unwrap();

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(ui_res.render_pass)
            .framebuffer(ui_sc.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D{width: swapchain.width, height: swapchain.height },
            })
            .clear_values(&[vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.105, 0.105, 0.105, 1.0],
                },
            }]);

        unsafe {

        // RP will transition from TRANSFER_DST -> PRESENT_SRC_KHR for us
        device.cmd_begin_render_pass(frame.cmd_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);

        device.cmd_bind_pipeline(frame.cmd_buffer, vk::PipelineBindPoint::GRAPHICS, ui_res.pipeline.vk_pipeline);

        device.cmd_set_viewport(
            frame.cmd_buffer,
            0,
            &[vk::Viewport {
                width: swapchain.width as f32,
                height: swapchain.height as f32,
                max_depth: 1.0,
                ..Default::default()
            }],
        );

        device.cmd_bind_index_buffer( frame.cmd_buffer, ui_sc.index_buffer.vk, 0, vk::IndexType::UINT32);
        device.cmd_bind_vertex_buffers(frame.cmd_buffer, 0, &[ui_sc.vertex_buffer.vk], &[0]);
        device.cmd_bind_descriptor_sets(frame.cmd_buffer, vk::PipelineBindPoint::GRAPHICS, ui_res.pipeline.layout.vk, 0, &[ui_res.descriptor_set], &[],);

        }

        // Ortho projection
        let projection = orthographic_vk( 0.0,
            swapchain.width as f32/pixels_per_point,
            0.0,
            -(swapchain.height as f32/pixels_per_point),
            -1.0,
            1.0,
        );

        unsafe {
        let projection = any_as_u8_slice(&projection);
        device.cmd_push_constants( frame.cmd_buffer, ui_res.pipeline.layout.vk, vk::ShaderStageFlags::VERTEX, 0, projection);
        }

        self.record_ui_drawing(&primitives, pixels_per_point);

        unsafe { device.cmd_end_render_pass(frame.cmd_buffer) };
    }

    pub fn record_pipeline_graph(&mut self) {
        let device = &self.vk_core.device;
        let graph_frame = &self.graph.frames[self.frame_index];
        let frame = &mut self.frames[self.frame_index];

        unsafe {
        device.cmd_reset_query_pool(frame.cmd_buffer,
                                    frame.timer.query_pool,
                                    0, // first-query-idx
                                    frame.timer.query_pool_size);

        if self.graph.is_compute() {
            command::transition_image_layout(&device, frame.cmd_buffer, graph_frame.get_output_image(), vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
        }

        command::execute_pipeline_graph(&device, frame, graph_frame, &self.graph);

        let frame = &self.frames[self.frame_index];

        if self.graph.is_compute() {
            command::transition_image_layout(&device, frame.cmd_buffer, graph_frame.get_output_image(), vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        }

        }
    }

    pub fn record_swapchain_blit(&self) {
        let swapchain_image = if self.swapchain.is_some() { Some(self.get_swapchain().images[self.present_index as usize]) } else { None };
        let device = &self.vk_core.device;
        let graph_frame = &self.graph.frames[self.frame_index];
        let frame = &self.frames[self.frame_index];

        if let (Some(swapchain), Some(swapchain_image)) = (self.swapchain.as_ref(), swapchain_image) {
            command::transition_image_layout(&device, frame.cmd_buffer, swapchain_image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

            let src_width  = if self.info.has_input_image { self.info.width  } else { swapchain.width };
            let src_height = if self.info.has_input_image { self.info.height } else { swapchain.height };
            let dst_width  = swapchain.width;
            let dst_height = swapchain.height;
            
            // All 0
            let color = vk::ClearColorValue::default();
            let subres_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            };

            // If we're partial blitting, clear the rest of the frame so the UI
            // doesn't drag around the borders of it
            if (src_width, src_height) != (dst_width, dst_height) {
                unsafe {
                device.cmd_clear_color_image(frame.cmd_buffer, swapchain_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &color, &[subres_range]);
                }
            }

            /* TODO?: Currently, we are using blit_image because it will do the format
             * conversion for us. However, another alternative is to do copy_image
             * after specifying th final compute shader destination image as the same
             * format as the swapchain format. Maybe worth measuring perf difference later */
            command::blit_copy(device, frame.cmd_buffer, &command::BlitCopy {
                src_width, src_height,
                dst_width, dst_height,
                src_image: graph_frame.get_output_image(),
                dst_image: swapchain_image,
                src_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                center: true
            });
        }
    }


    pub fn record(&mut self) {
        self.record_pipeline_graph();
        self.record_swapchain_blit();

        if self.ui.is_some() {
            self.record_ui();
        }
        else {
            let frame = &self.frames[self.frame_index];
            let device = &self.vk_core.device;

            // we blit to swapchain from compute and present directly
            self.swapchain.as_ref().map(|sc| {
                let swapchain_image = sc.images[self.present_index as usize];
                command::transition_image_layout(&device, frame.cmd_buffer, swapchain_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
            });
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

        let mut submit_info = vk::SubmitInfo::default()
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
            &[submit_info],
            frame.fence,
        ).expect("queue submit failed.");
        }

        if self.swapchain.is_some() {
            let wait_semaphores = [frame.render_complete_semaphore];
            let swapchain = self.get_swapchain();
            let swapchains = [swapchain.vk];
            let image_indices = [self.present_index];
            let present_info = vk::PresentInfoKHR::default()
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

    pub fn trigger_reloads(&mut self) -> Result<bool> {

        // If the window has changed, we need to reload the swapchain
        // Normally, resize_swapchain() is called explicitly elsewhere,
        // so this should only be the case if we got an OUT_OF_DATE KHR or suboptimal present
        if self.swapchain_rebuilt_required {
            self.resize_swapchain(self.window_width, self.window_height)?;
        }

        let mut full_reload_performed = false;

        // If our configuration has changed, live reload it
        if let Some(reload_config) = &mut self.reload_config {
            // Swap reload into info
            {
            std::mem::swap(&mut self.info.config, reload_config);
            }

            // Try to reload with the new config
            full_reload_performed = self.recreate_graph().is_ok();

            // If the reload failed, return to our original state
            // If any of our shaders have changed, live reload them
            if full_reload_performed {
                self.last_modified_shader_times.clear();
            }
            else {
                let reload_config = self.reload_config.as_mut().context("reload_config invalid")?;
                std::mem::swap(&mut self.info.config, reload_config);
            }

            self.reload_config = None;
        }

        self.reload_changed_pipelines();

        Ok(full_reload_performed)
    }

    /*
    pub fn last_frame_gpu_times(&mut self) -> String {
        self.frames[self.frame_index].timer.get_elapsed_ms()
    }
    */
    pub fn resize_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe {
        self.vk_core.device.device_wait_idle()?;

        self.window_width  = width;
        self.window_height = height;

        self.swapchain.as_mut().context("No swapchain previously created")?.rebuild(&self.vk_core, self.window_width, self.window_height)?;

        // Recreate per-swapchain resources
        self.per_sc_res = self.ui_res.as_ref().map(|ui|  {
            let swapchain = self.swapchain.as_ref().unwrap();

            let sc_res: Result<Vec<PerSwapChainRes>> = swapchain.views.iter().map(|sc_image_view|{
                PerSwapChainRes::new(&self.vk_core, ui.render_pass, *sc_image_view, swapchain.width, swapchain.height)
            }).collect();

            sc_res
        }).transpose()?;

        if !self.info.has_input_image {
            // Pipeline dimensions shouldn't change if we are using an input
            // We just blit the same dimensions to a larger window
            self.recreate_graph()?;

            // For generative shaders where we want the blit size = window size
            self.info.width  = self.window_width;
            self.info.height = self.window_height;
        }
        }

        self.outdate_frames();
        self.swapchain_rebuilt_required = false;

        Ok(())
    }

    pub fn new(info: RenderInfo, window: Option<&Window>) -> Result<Render> {
        unsafe {
        let vk_core = VkCore::new(window)?;

        let graph = Self::create_graph(&vk_core, &info)?;

        // We use rgba8 as the input file format
        let buffer_size = (info.width as vk::DeviceSize)*(info.height as vk::DeviceSize)*4;

        // This staging buffer will be used to transfer the original input file into an mimage
        let staging_buffer = vkutils::create_buffer(&vk_core,
                                                        "input-image-staging-buffer".to_string(),
                                                        buffer_size,
                                                        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                                                        gpu_alloc::MemoryLocation::GpuToCpu);

        let staging_srgb_image = vkutils::create_image(&vk_core,
                                                       "input-image-srgb".to_string(),
                                                       vk::Format::R8G8B8A8_SRGB,
                                                       info.width, info.height);

        let last_modified_shader_times: HashMap<String, u64> = utils::get_modified_times(&graph.pipelines);


        let (window_width, window_height) = if window.is_some() { (info.width, info.height) } else { (0, 0 ) };

        let swapchain = window.map(|_| SwapChain::new(&vk_core, info.width, info.height)).transpose()?;
        let ui = window.map(|w| ui::Ui::new(w));
        let ui_res = swapchain.as_ref().map(|sc| UiResources::new(&vk_core.device, sc.surface_format.format)).transpose()?;

        let frames: Result<Vec<Frame>, _> =
            (0..info.num_frames).map(|_|{
            Frame::new(&vk_core, info.config.graph_pipelines.len() as u32)
        }).collect();

        let per_swapchain: Option<Vec<PerSwapChainRes>> =  ui_res.as_ref().map(|ui|  {
            let swapchain = swapchain.as_ref().unwrap();

            let sc_res: Result<Vec<PerSwapChainRes>> = swapchain.views.iter().map(|sc_image_view|{
                PerSwapChainRes::new(&vk_core, ui.render_pass, *sc_image_view, swapchain.width, swapchain.height)
            }).collect();

            sc_res
        }).transpose()?;


        Ok(Render {
            frames: frames?,
            frame_outdated: (0..info.num_frames).map(|_| { true } ).collect(),
            graph,
            info,
            staging_srgb_image,
            staging_buffer,
            last_modified_shader_times,
            present_index: 0,
            frame_index: 0,
            vk_core,
            swapchain,
            swapchain_rebuilt_required: false,
            pipeline_buffer_data: HashMap::new(),
            reload_config: None,
            window_width,
            window_height,
            ui,
            ui_res,
            per_sc_res: per_swapchain
        })

        }
    }
}

impl Drop for UiResources {
    fn drop(&mut self) {
        unsafe {
            let device = &self.pipeline.device;
            device.destroy_descriptor_pool(self._descriptor_pool, None);
        }
    }
}

impl Drop for UiPerSwapchain {
    fn drop(&mut self) {
        unsafe {
            let device = &self.vertex_buffer.device;
            device.destroy_framebuffer(self.framebuffer, None);
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
