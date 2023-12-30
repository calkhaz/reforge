extern crate ash;
extern crate shaderc;
extern crate gpu_allocator;

use ash::vk;
use spirv_reflect::types::{ReflectDescriptorBinding, ReflectDescriptorType, ReflectBlockVariable, ReflectTypeFlags};
use std::collections::HashSet;
use std::ffi::CStr;
use std::default::Default;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use std::ops::Drop;

use crate::vulkan::core::VkCore;
use crate::vulkan::vkutils;
use crate::vulkan::vkutils::{Buffer, Image, Sampler};
use crate::vulkan::shader::Shader;
use crate::warnln;

pub const FILE_INPUT: &str = "rf:file-input";
pub const SWAPCHAIN_OUTPUT: &str = "rf:swapchain";

pub struct PipelineLayout {
    pub vk: vk::PipelineLayout,
    pub descriptor_layout: vk::DescriptorSetLayout
}

#[derive(Debug, Clone)]
pub struct PipelineInfo {
    pub name: String,
    pub shader: Rc<RefCell<Shader>>,
    pub input_images: Vec<(String, ReflectDescriptorBinding)>,
    pub output_images: Vec<(String, ReflectDescriptorBinding)>,
    pub input_ssbos: Vec<(String, ReflectDescriptorBinding)>,
    pub output_ssbos: Vec<(String, ReflectDescriptorBinding)>,
}

pub struct Pipeline {
    device: Rc<ash::Device>,
    pub name: String,
    pub info: PipelineInfo,
    pub layout: PipelineLayout,
    pub vk_pipeline: ash::vk::Pipeline,

    // In the case of a graphics pipeline
    vertex_shader: Option<Shader>,
    pub render_pass: Option<vk::RenderPass>
}

pub struct PipelineGraphFrame {
    device: Rc<ash::Device>,
    pub images: HashMap<String, Image>,
    pub buffers: HashMap<String, Buffer>,
    pub ubos: HashMap<String, HashMap<String, BufferBlock>>,
    pub descriptor_sets: HashMap<String, vk::DescriptorSet>,
    pub attachment_image: Option<Image>,
    pub framebuffer: Option<vk::Framebuffer>,
    image_reuse_remapping: HashMap<String, String>
}

pub struct PipelineGraphInfo {
    pub pipeline_infos: HashMap<String, PipelineInfo>,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub num_frames: usize
}

pub struct PipelineGraph {
    /* The sorted execution of the graph
     * Command buffers can process this order */
    pub ordered_pipelines: Vec<Vec<Rc<RefCell<Pipeline>>>>,
    device: Rc<ash::Device>,
    pub frames: Vec<PipelineGraphFrame>,
    pub width: u32,
    pub height: u32,
    pub pipelines: HashMap<String, Rc<RefCell<Pipeline>>>,
    _sampler: Sampler, // Stored here so it doesn't get dropped
    descriptor_pool: vk::DescriptorPool,
    pub bind_point: vk::PipelineBindPoint
}

struct PipelineGraphFrameInfo<'a> {
    pub ordered_pipelines: &'a Vec<Vec<Rc<RefCell<Pipeline>>>>,
    pub descriptor_pool: vk::DescriptorPool,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub sampler: &'a Sampler,
    image_reuse_remapping: &'a HashMap<String, String>
}

pub struct BufferBlock {
    pub size: u32,
    pub offset:u32,
    pub block_type: ReflectTypeFlags,
    pub buffer: Rc<Buffer>
}

fn image_remap_name<'a>(name: &'a String, mapping: &'a HashMap<String, String>) -> &'a String {
    match mapping.get(name) {
        Some(n) => image_remap_name(n, mapping), None => name
    }
}

impl PipelineGraphFrame {
    pub fn get_input_image(&self) -> &Image {
        self.images.get(FILE_INPUT).unwrap()
    }

    pub fn get_output_image(&self) -> vk::Image {
        match &self.attachment_image {
            Some(image) => image.vk,
            None => {
                let swapchain_output = String::from(SWAPCHAIN_OUTPUT);
                let name = image_remap_name(&swapchain_output, &self.image_reuse_remapping);
                self.images.get(name).unwrap().vk
            }
        }
    }

    unsafe fn image_write(image: &Image, infos: &mut Vec<vk::DescriptorImageInfo>, binding: &ReflectDescriptorBinding, set: vk::DescriptorSet, sampler: &Sampler) -> vk::WriteDescriptorSet {
        infos.push(vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: image.view.unwrap(),
            sampler: sampler.vk
        });

        vk::WriteDescriptorSet {
            dst_set: set,
            dst_binding: binding.binding,
            descriptor_count: 1,
            descriptor_type: vkutils::reflect_desc_to_vk(binding.descriptor_type).unwrap(),
            p_image_info: infos.last().unwrap(),
            ..Default::default()
        }
    }

    unsafe fn buffer_write(buffer: &Buffer, desc_type: vk::DescriptorType, infos: &mut Vec<vk::DescriptorBufferInfo>, binding_idx: u32, set: vk::DescriptorSet) -> vk::WriteDescriptorSet {
        // TODO: If we get a different allocator, we'll want to change the offset and range here
        infos.push(vk::DescriptorBufferInfo {
            buffer: buffer.vk,
            offset: 0,
            range : ash::vk::WHOLE_SIZE
        });

        vk::WriteDescriptorSet {
            dst_set: set,
            dst_binding: binding_idx,
            descriptor_count: 1,
            descriptor_type: desc_type,
            p_buffer_info: infos.last().unwrap(),
            ..Default::default()
        }
    }

    unsafe fn build_framebuffer(core: &VkCore, image: &Image, render_pass: vk::RenderPass, width: u32, height: u32) -> vk::Framebuffer {
        let info = vk::FramebufferCreateInfo {
            render_pass,
            attachment_count: 1,
            p_attachments: &image.view.unwrap(),
            width,
            height,
            layers: 1,
            ..Default::default()
        };

        core.device.create_framebuffer(&info, None).unwrap_or_else(|err| panic!("Error: {}", err))
    }

    unsafe fn new(core: &VkCore, frame_info: &PipelineGraphFrameInfo) -> PipelineGraphFrame {
        let device = &core.device;
        let format = frame_info.format;

        // Create per-frame images, descriptor sets
        let mut images: HashMap<String, Image> = HashMap::new();
        let mut buffers: HashMap<String, Buffer> = HashMap::new();
        let mut descriptor_sets: HashMap<String, vk::DescriptorSet> = HashMap::new();

        let mut ssbo_sizes: HashMap<String, u32> = HashMap::new();
        // Pipeline -> <buffer-name -> block/buffer>
        let mut ubos: HashMap<String, HashMap<String, BufferBlock>> = HashMap::new();

        let mut attachment_image: Option<Image> = None;
        let mut framebuffer: Option<vk::Framebuffer> = None;

        for layer in frame_info.ordered_pipelines {
            for pipeline in layer {
                if let Some(render_pass) = pipeline.borrow().render_pass {
                    attachment_image = Some(vkutils::create_image(core, "color-attachment".to_string(), format, frame_info.width, frame_info.height));
                    framebuffer = Some(Self::build_framebuffer(core, &attachment_image.as_ref().unwrap(), render_pass, frame_info.width, frame_info.height));
                }
            }
        }

        for layer in frame_info.ordered_pipelines {
            for pipeline in layer {
                let info = &pipeline.borrow().info;
                let mut add_ssbo_sizes = |buffer_name_pairs: &Vec<(String, ReflectDescriptorBinding)>| {
                    for (buffer_name, binding) in buffer_name_pairs {
                        let size: u32 = binding.block.members.iter().map(|s| s.padded_size).sum();

                        // Insert the size or max of current size and new size found
                        ssbo_sizes.entry(buffer_name.clone()).and_modify(|curr_val| {
                            *curr_val = std::cmp::max(size, *curr_val);
                        }).or_insert(size);
                    }
                };

                add_ssbo_sizes(&info.input_ssbos);
                add_ssbo_sizes(&info.output_ssbos);
            }
        }

        let mut buffer_remap: HashMap<String, String> = HashMap::new();
        for layer in frame_info.ordered_pipelines {
            for pipeline in layer {
                let name = &pipeline.borrow().name;
                let info = &pipeline.borrow().info;
                let layout_info = &[pipeline.borrow().layout.descriptor_layout];

                let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(frame_info.descriptor_pool)
                    .set_layouts(layout_info);

                let descriptor_set = device
                    .allocate_descriptor_sets(&desc_alloc_info)
                    .unwrap()[0];

                let mut desc_image_infos: Vec<vk::DescriptorImageInfo> =
                    Vec::with_capacity(info.input_images.len() + info.output_images.len());

                let mut desc_buffer_infos: Vec<vk::DescriptorBufferInfo> =
                    Vec::with_capacity(info.shader.borrow().bindings.buffers.len());

                let mut descriptor_writes: Vec<vk::WriteDescriptorSet> =
                    Vec::with_capacity(info.input_images.len()  + info.output_images .len() +
                                       info.shader.borrow().bindings.buffers.len());

                // The only input not guaranteed to appear in any outputs is a file input
                // Go ahead and create it if it is found
                for (image_name, _) in &info.input_images {
                    // We only want one FILE_INPUT input image across frames as it will never change
                    if image_name == FILE_INPUT {
                        images.entry(image_name.clone()).or_insert(
                            vkutils::create_image(core, name.clone(), format, frame_info.width, frame_info.height)
                        );
                    }
                }

                // Create output descriptor writes and create images
                for (image_name, binding) in &info.output_images {
                    // Reuse images when possible
                    let name = image_remap_name(image_name, frame_info.image_reuse_remapping);

                    let image = images.entry(name.clone()).or_insert(
                        vkutils::create_image(core, name.clone(), format, frame_info.width, frame_info.height)
                    );

                    descriptor_writes.push(Self::image_write(&image, &mut desc_image_infos, binding, descriptor_set, frame_info.sampler));
                }

                // Create input image descriptor writes
                for (image_name, binding) in &info.input_images {
                    let mut found_point_op = false;
                    
                    for (_, output_binding) in &info.output_images {
                        if output_binding.binding == binding.binding { found_point_op = true; }
                    }

                    if found_point_op { continue; }

                    // Reuse images when possible
                    let name = image_remap_name(image_name, frame_info.image_reuse_remapping);

                    // Besides file inputs, input images are a subset of output images, so find them and bind the descriptor
                    let image = images.get(name).unwrap_or_else(|| panic!("No image found for input {}", name));
                    descriptor_writes.push(Self::image_write(&image, &mut desc_image_infos, binding, descriptor_set, frame_info.sampler));
                }

                for (output_name, output_binding) in &info.output_ssbos {
                    for (input_name, input_binding) in &info.input_ssbos {
                        if input_binding.binding == output_binding.binding {
                            buffer_remap.insert(output_name.clone(), input_name.clone());
                        }
                    }
                }


                let buffer_remap_inv: HashMap<String, String> = buffer_remap
                    .iter()
                    .fold(HashMap::new(), |mut acc, (k, v)| {
                        acc.insert(v.clone(), k.clone());
                        acc
                    });

                // Create output descriptor writes and create ssbo buffers
                for (buffer_name, binding) in &info.output_ssbos {
                    let binding_idx = binding.binding;
                    let name = image_remap_name(buffer_name, &buffer_remap);

                    let buffer = buffers.entry(name.clone()).or_insert({
                        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
                        let size = *ssbo_sizes.get(name).unwrap();
                        vkutils::create_buffer(core, name.to_string(), size as u64, usage, gpu_allocator::MemoryLocation::CpuToGpu)
                    });

                    descriptor_writes.push(Self::buffer_write(&buffer, vk::DescriptorType::STORAGE_BUFFER, &mut desc_buffer_infos, binding_idx, descriptor_set));
                }

                // Create input buffer descriptor writes
                for (buffer_name, binding) in &info.input_ssbos {
                    //if buffer_remap_inv.contains_key(buffer_name) {
                    //    continue;
                    //}

                    let mut found_point_op = false;
                    
                    for (_, output_binding) in &info.output_ssbos {
                        if output_binding.binding == binding.binding { found_point_op = true; }
                    }

                    if found_point_op { continue; }

                    let name = image_remap_name(buffer_name, &buffer_remap);
                    let binding_idx = binding.binding;
                    let buffer = buffers.get(name).unwrap_or_else(|| panic!("No buffer found for input {}", buffer_name));
                    descriptor_writes.push(Self::buffer_write(&buffer, vk::DescriptorType::STORAGE_BUFFER, &mut desc_buffer_infos, binding_idx, descriptor_set));
                }

                // ubos for this pipeline
                let mut pipeline_ubos: HashMap<String, BufferBlock> = HashMap::new();

                fn recurse_block(reflect_block: &ReflectBlockVariable, base_name: &String, buffer: &Rc<Buffer>, ubos: &mut HashMap<String, BufferBlock>) { // -> BufferBlock
                    let block = BufferBlock {
                        size: reflect_block.size,
                        offset: reflect_block.offset,
                        block_type: reflect_block.type_description.as_ref().unwrap().type_flags,
                        buffer: Rc::clone(buffer),
                    };

                    let name = if base_name.is_empty() { reflect_block.name.clone() }
                    else                               { format!("{}.{}", base_name, reflect_block.name.clone()) };

                    if !reflect_block.name.is_empty() {
                        ubos.insert(name.clone(), block);
                    }

                    reflect_block.members.iter().for_each(|member| recurse_block(&member, &name, &buffer, ubos));
                }

                for buffer_reflect in &info.shader.borrow().bindings.buffers {
                    if buffer_reflect.descriptor_type == ReflectDescriptorType::UniformBuffer {
                        let buffer_name = format!("{}:{}", pipeline.borrow().name, buffer_reflect.type_description.as_ref().unwrap().type_name.clone());
                        let usage = vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;

                        let buffer = Rc::new(vkutils::create_buffer(core, buffer_name, buffer_reflect.block.size as u64, usage, gpu_allocator::MemoryLocation::CpuToGpu));
                        descriptor_writes.push(Self::buffer_write(&buffer, vk::DescriptorType::UNIFORM_BUFFER, &mut desc_buffer_infos, buffer_reflect.binding, descriptor_set));

                        recurse_block(&buffer_reflect.block, &"".to_string(), &buffer, &mut pipeline_ubos);
                    }
                }

                ubos.insert(name.clone(), pipeline_ubos);

                device.update_descriptor_sets(&descriptor_writes, &[]);
                descriptor_sets.insert(name.to_string(), descriptor_set);
            }
        }

        PipelineGraphFrame {
            device: Rc::clone(&device),
            images,
            buffers,
            ubos,
            descriptor_sets,
            attachment_image,
            framebuffer,
            image_reuse_remapping: frame_info.image_reuse_remapping.clone()
        }
    }
}

impl PipelineGraph {
    pub fn is_compute (&self) -> bool { self.bind_point == vk::PipelineBindPoint::COMPUTE }

    pub unsafe fn rebuild_pipeline(&mut self, name: &str) {
        let device = &self.device;
        let is_compute = self.is_compute();
        let mut pipeline = self.pipelines.get_mut(name).unwrap().borrow_mut();

        if pipeline.info.shader.borrow().path.is_none() {
            return;
        }

        let shader = Shader::from_path(&device, &pipeline.info.shader.borrow().path.as_ref().unwrap());

        if let Some(shader) = shader { 
            let vk_pipeline = if is_compute {
                Self::build_vk_compute_pipeline(&device,
                                                shader.module,
                                                pipeline.layout.vk)
            }
            else {
                Self::build_vk_graphics_pipeline(&device,
                                                 self.width,
                                                 self.height,
                                                 pipeline.vertex_shader.as_ref().unwrap().module,
                                                 shader.module,
                                                 pipeline.layout.vk,
                                                 pipeline.render_pass.unwrap())
            };

            // In some cases, the spirv code compiles but we fail to create the pipeline due to
            // other issues, which can still be related to the shader code being wrong or
            // incompatible with current configurations
            match vk_pipeline {
                Ok(vk_pipeline) =>  {
                    destroy_pipeline(&mut *pipeline, false);
                    pipeline.vk_pipeline = vk_pipeline;
                    pipeline.info.shader.borrow_mut().module = shader.module;
                },
                Err(error) => {
                    warnln!("{:?}", error);
                }
            }
        }
    }

    pub fn map_to_desc_size(map: &HashMap<vk::DescriptorType, u32>) -> Vec<vk::DescriptorPoolSize> {
        let mut sizes: Vec<vk::DescriptorPoolSize> = Vec::with_capacity(map.keys().len());

        for (desc_type, count) in map {
            sizes.push(vk::DescriptorPoolSize {
                ty: *desc_type,
                descriptor_count: *count
            });
        }

        sizes
    }

    pub unsafe fn build_pipeline_layout(device: Rc<ash::Device>,
                                        info: &PipelineInfo,
                                        pool_sizes: &mut HashMap<vk::DescriptorType, u32>,
                                        num_frames: usize) -> PipelineLayout {
        // create descriptor layouts, add descriptor pool sizes, and add pipelines to hashmap
        let layout_bindings = vkutils::create_descriptor_layout_bindings(&info.shader.borrow().bindings, num_frames, pool_sizes);

        let descriptor_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);
        let descriptor_layout = [device
            .create_descriptor_set_layout(&descriptor_info, None)
            .unwrap()];

        let pipeline_layout = device.
            create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&descriptor_layout), None).unwrap();

        PipelineLayout {
            vk: pipeline_layout,
            descriptor_layout: descriptor_layout[0]
        }
    }

    pub unsafe fn build_vk_compute_pipeline(device: &ash::Device,
                                            shader: vk::ShaderModule,
                                            pipeline_layout: vk::PipelineLayout) -> Result<vk::Pipeline, String> {


        let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");
        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo {
            module: shader,
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        };

        let vk_info = vk::ComputePipelineCreateInfo {
            layout: pipeline_layout,
            stage: shader_stage_create_infos,
            ..Default::default()
        };

        match device.create_compute_pipelines(vk::PipelineCache::null(), &[vk_info], None) {
            Ok(pipelines) => {
                Ok(pipelines[0])
            },
            Err(err) => {
                Err(format!("Failed to create graphics pipeline: {:?}", err))
            }
        }

    }

    unsafe fn build_vertex_shader(device: &ash::Device) -> Shader {
        // full-screen triangle
        let vertex_shader_code = r#"
            #version 450

            vec2 positions[3] = vec2[](
                vec2(-1.0, -3.0), // top left
                vec2(-1.0,  1.0), // bottom left
                vec2( 3.0 , 1.0)  // bottom right
            );

            void main() {
                gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
            }
        "#;

        Shader::from_contents(&device, "full-screen-triangle".to_string(), vk::ShaderStageFlags::VERTEX, vertex_shader_code.to_string()).unwrap()
    }

    pub unsafe fn build_vk_graphics_pipeline(device: &ash::Device,
                                             width: u32,
                                             height: u32,
                                             vertex_shader: vk::ShaderModule,
                                             fragment_shader: vk::ShaderModule,
                                             pipeline_layout: vk::PipelineLayout,
                                             render_pass: vk::RenderPass) -> Result<vk::Pipeline, String> {

        let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");

        let shader_stages = vec![
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vertex_shader,
                p_name: shader_entry_name.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: fragment_shader,
                p_name: shader_entry_name.as_ptr(),
                ..Default::default()
            }
        ];

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            line_width: 1.0,
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            ..Default::default()
        };

        let scissors = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height}
        };

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            max_depth: 1.0,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            p_viewports: &viewport,
            p_scissors: &scissors,
            ..Default::default()
        };

        let blend_attachment = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            color_write_mask: vk::ColorComponentFlags::RGBA,
            ..Default::default()
        };

        let blend_state = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: vk::FALSE,
            attachment_count: 1,
            p_attachments: &blend_attachment,
            ..Default::default()
        };

        let vk_info = vk::GraphicsPipelineCreateInfo {
            render_pass,
            layout: pipeline_layout,
            stage_count: 2,
            p_stages: shader_stages.as_ptr(),
            subpass: 0,
            p_input_assembly_state: &input_assembly,
            p_rasterization_state: &rasterization_state,
            p_viewport_state: &viewport_state,
            p_color_blend_state: &blend_state,
            p_vertex_input_state: &vertex_input_state,
            ..Default::default()
        };

        match device.create_graphics_pipelines(vk::PipelineCache::null(), &[vk_info], None) {
            Ok(pipelines) => {
                Ok(pipelines[0])
            },
            Err(err) => {
                Err(format!("Failed to create graphics pipeline: {:?}", err))
            }
        }
    }

    pub unsafe fn build_render_pass(device: &ash::Device, format: vk::Format) -> vk::RenderPass {
        let color_attachment = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        };

        let attachment_desc = vk::AttachmentDescription {
            format: format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            //initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ..Default::default()
        };

        let subpass_desc = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment,
            ..Default::default()
        };

        let info = vk::RenderPassCreateInfo {
            attachment_count: 1,
            p_attachments: &attachment_desc,
            subpass_count: 1,
            p_subpasses: &subpass_desc,
            ..Default::default()
        };

        device.create_render_pass(&info, None).unwrap_or_else(|err| panic!("Error: {}", err))
    }

    fn reusable_image_remapping(ordered_pipelines: &Vec<Vec<PipelineInfo>>) -> HashMap<String, String> {
        let mut free_images : Vec<String> = Vec::new();
        let mut images: HashSet<String> = HashSet::new();
        let mut image_reuse: HashMap<String, String> = HashMap::new();

        let node_uses_image = |node: &PipelineInfo, image_name: &String| -> bool {
            if node.input_images .iter().any(|n| n.0 == *image_name) { return true; }
            if node.output_images.iter().any(|n| n.0 == *image_name) { return true; }
            false
        };

        let image_still_in_use = |name: &String, start_layer: usize, image_reuse: &HashMap<String, String>| -> bool {
            for execution_layers in ordered_pipelines.iter().skip(start_layer) {
                for node in execution_layers {

                    // Found the actual allocation still in use
                    if node_uses_image(node, name) { return true; }

                    // Found a remap to the actual allocation as an input
                    for (input_image, _) in &node.input_images {
                        if let Some(reused_image) = image_reuse.get(input_image) {
                            if reused_image == name { return true; }
                        }
                    }
                    // Found a remap to the actual allocation as an output
                    for (output_image, _) in &node.output_images {
                        if let Some(reused_image) = image_reuse.get(output_image) {
                            if reused_image == name { return true; }
                        }
                    }
                }
            }

            false
        };

        for (layer_idx, execution_layers) in ordered_pipelines.iter().enumerate() {
            for name in &images {
                // The image is already known to be free
                if free_images.contains(name) { continue; }

                // Free the image if it is no longer used
                if !image_still_in_use(name, layer_idx, &image_reuse) {
                    free_images.push(name.clone());
                }
            }

            execution_layers.iter().for_each(|node| {
                for (image_name, output_binding) in &node.output_images {

                    // could we have already remapped the input ? i don't think so because it is
                    // needed for this node
                    
                    let mut found_point_op = false;
                    for (input_name, input_binding) in &node.input_images {
                        if output_binding.binding == input_binding.binding {
                            found_point_op = true;
                            image_reuse.insert(image_name.clone(), input_name.clone());
                        }
                    }
                    if found_point_op { continue }
                    if image_name == SWAPCHAIN_OUTPUT { continue };

                    // allocate if there is no free image
                    if free_images.is_empty() {
                        images.insert(image_name.clone());
                    }
                    // remap and reuse
                    else {
                        let remap_name = free_images.pop().unwrap();
                        image_reuse.insert(image_name.clone(), remap_name);
                    }
                }
            });
        }

        image_reuse
    }

    fn order_by_execution(infos: &HashMap<String, PipelineInfo>) -> Option<Vec<Vec<PipelineInfo>>> {
        let mut ordered_pipelines: Vec<Vec<PipelineInfo>> = Vec::new();
        let mut unexecuted_nodes_set: HashSet<String> = infos.keys().cloned().collect();

        // Function to search other other for what pipelines output into the given pipeline
        let get_input_node_names = |info: &PipelineInfo| -> Vec<String> {
            let mut nodes: Vec<String> = Vec::new();

            let input_images: Vec<String> = info.input_images.iter().map(|name| name.0.clone()).collect();
            let input_ssbos : Vec<String> = info.input_ssbos .iter().map(|name| name.0.clone()).collect();

            for (candidate_name, candidate_pipeline) in infos {
                // Check if any of the outputs are used as inputs for the given pipeline
                let output_has_searched_input_image = candidate_pipeline.output_images.iter().any(|name| input_images.contains(&name.0));
                let output_has_searched_input_ssbo  = candidate_pipeline.output_ssbos .iter().any(|name| input_ssbos .contains(&name.0));

                if output_has_searched_input_image || output_has_searched_input_ssbo {
                    nodes.push(candidate_name.clone());
                }
            }

            nodes
        };

        /* To flatten the graph, we want to visit all the nodes and check if any of them
         * are able to be executed
         *
         * The ready status is determined by checking a node's inputs
         * to see if they have already been executed
         *
         * If we are able to add them as the next action to execute 
         * and then put a barrier after the new group of executed nodes
         *
         * Ex: 
         * input  -> gaussian    -> combination:input_image0
         * output -> edge_detect -> combination:input_image1
         * combination -> output
         *
         * Result:
         * gaussian + edge_detect -> barrier -> combination */
        while !unexecuted_nodes_set.is_empty() {
            let unexecuted_nodes: Vec<String> = unexecuted_nodes_set.iter().cloned().collect();

            let mut node_infos: Vec<PipelineInfo> = Vec::new();

            for node in &unexecuted_nodes {
                let pipeline = infos.get(node).unwrap();
                let input_nodes = get_input_node_names(pipeline);

                // If all input dependencies have been executed, this node is ready to execute
                let ready_to_execute = !input_nodes.iter().any(|n| unexecuted_nodes.contains(n));

                if ready_to_execute {
                    unexecuted_nodes_set.remove(node);
                    node_infos.push(pipeline.clone());
                }
            }

            if unexecuted_nodes.len() == unexecuted_nodes_set.len() {
                warnln!("Graph incorrectly constructed. Failed to add nodes into execution: {:?}", unexecuted_nodes);
                return None
            }

            ordered_pipelines.push(node_infos);
        }


        Some(ordered_pipelines)
    }

    pub unsafe fn new(core: &VkCore, gi: PipelineGraphInfo) -> Option<PipelineGraph> {
        let ordered_infos = Self::order_by_execution(&gi.pipeline_infos)?;
        let image_reuse_remapping = Self::reusable_image_remapping(&ordered_infos);

        // Track descriptor pool sizes by descriptor type
        let mut pool_sizes: HashMap<vk::DescriptorType, u32> = HashMap::new();
        let mut pipelines: HashMap<String, Rc<RefCell<Pipeline>>> = HashMap::new();
        let mut ordered_pipelines: Vec<Vec<Rc<RefCell<Pipeline>>>> = Vec::new();
        let mut bind_point = vk::PipelineBindPoint::COMPUTE;

        for layer in ordered_infos {
            let mut pipeline_layer: Vec<Rc<RefCell<Pipeline>>> = Vec::new();

            for info in layer {
                let name = info.name.clone();

                let pipeline_layout = Self::build_pipeline_layout(Rc::clone(&core.device), &info, &mut pool_sizes, gi.num_frames);

                if info.shader.borrow().stage == vk::ShaderStageFlags::FRAGMENT {
                    bind_point = vk::PipelineBindPoint::GRAPHICS;
                    assert!(pipelines.len() < 1, "Can only have one pipeline when using fragment shaders");
                    let render_pass = Some(Self::build_render_pass(&core.device, gi.format));
                    let vertex_shader = Some(Self::build_vertex_shader(&core.device));
                    let vk_pipeline = Self::build_vk_graphics_pipeline(&core.device,
                                                                       gi.width,
                                                                       gi.height,
                                                                       vertex_shader.as_ref().unwrap().module,
                                                                       info.shader.borrow().module,
                                                                       pipeline_layout.vk,
                                                                       render_pass.unwrap()).
                                                                       unwrap_or_else(|err| panic!("Error: {}", err));

                    let pipeline = Rc::new(RefCell::new(Pipeline {
                        device: Rc::clone(&core.device),
                        name: name.clone(),
                        info,
                        layout: pipeline_layout,
                        vk_pipeline,
                        render_pass,
                        vertex_shader
                    }));

                    pipelines.insert(name.to_string(), pipeline.clone());
                    pipeline_layer.push(pipeline);
                }
                else { // COMPUTE
                    bind_point = vk::PipelineBindPoint::COMPUTE;
                    let vk_pipeline = Self::build_vk_compute_pipeline(&core.device,
                                                                      info.shader.borrow().module,
                                                                      pipeline_layout.vk).
                                                                      unwrap_or_else(|err| panic!("Error: {}", err));

                    let pipeline = Rc::new(RefCell::new(Pipeline {
                        device: Rc::clone(&core.device),
                        name: name.clone(),
                        info,
                        layout: pipeline_layout,
                        vk_pipeline,
                        render_pass: None,
                        vertex_shader: None
                    }));

                    pipelines.insert(name.to_string(), pipeline.clone());
                    pipeline_layer.push(pipeline);
                }
            }

            ordered_pipelines.push(pipeline_layer);
        }

        // Create descriptor pool
        let descriptor_size_vec = Self::map_to_desc_size(&pool_sizes);

        // We determine number of sets by the number of pipelines
        // Generally, each pipeline will have num_frames amount of descriptor copies
        // However, if there is a set of swapchain images being used for one pipeline, we will include that
        let num_max_sets = gi.num_frames as u32*pipelines.len() as u32;

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&descriptor_size_vec)
            .max_sets(num_max_sets);

        let descriptor_pool = core.device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap();

        let sampler = vkutils::create_sampler(Rc::clone(&core.device));

        let mut frames: Vec<PipelineGraphFrame> = Vec::with_capacity(gi.num_frames);

        for _ in 0..gi.num_frames {
            let graph_frame_info = PipelineGraphFrameInfo {
                ordered_pipelines: &ordered_pipelines,
                descriptor_pool: descriptor_pool,
                width: gi.width,
                height: gi.height,
                format: gi.format,
                sampler: &sampler,
                image_reuse_remapping: &image_reuse_remapping
            };

            frames.push(PipelineGraphFrame::new(core, &graph_frame_info));
        }

        Some(PipelineGraph {
            device: Rc::clone(&core.device),
            frames: frames,
            width: gi.width,
            height: gi.height,
            pipelines: pipelines,
            _sampler: sampler,
            descriptor_pool: descriptor_pool,
            ordered_pipelines,
            bind_point
        })
    }
}

fn destroy_pipeline(pipeline: &mut Pipeline, destroy_non_resizables: bool) {
    let device = &pipeline.device;

    unsafe {
    device.device_wait_idle().unwrap();
    device.destroy_pipeline(pipeline.vk_pipeline, None);
    if destroy_non_resizables {
        if let Some(render_pass) = pipeline.render_pass {
            device.destroy_render_pass(render_pass, None)
        }
        device.destroy_pipeline_layout(pipeline.layout.vk, None);
        device.destroy_descriptor_set_layout(pipeline.layout.descriptor_layout, None);

        if let Some(vertex_shader) = &pipeline.vertex_shader {
            device.destroy_shader_module(vertex_shader.module, None);
        }
    }
    device.destroy_shader_module(pipeline.info.shader.borrow().module, None);
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        destroy_pipeline(self, true);
    }
}

impl Drop for PipelineGraphFrame {
    fn drop(&mut self) {
        if let Some(framebuffer) = self.framebuffer {
            unsafe {
            self.device.destroy_framebuffer(framebuffer, None);
            }
        }
    }
}

impl Drop for PipelineGraph {
    fn drop(&mut self) {
        let device = &self.device;
        unsafe {

        self.device.device_wait_idle().unwrap();
        self.frames.clear();
        self.ordered_pipelines.clear();
        self.pipelines.clear();

        device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

