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

pub enum GraphAction {
    Pipeline(Rc<RefCell<Pipeline>>),
    Barrier
}

pub struct PipelineLayout {
    pub vk: vk::PipelineLayout,
    pub descriptor_layout: vk::DescriptorSetLayout
}

pub struct PipelineInfo {
    pub shader: Shader,
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
    pub vk_pipeline: ash::vk::Pipeline
}

pub struct PipelineGraphFrame {
    pub images: HashMap<String, Image>,
    pub buffers: HashMap<String, Buffer>,
    pub ubos: HashMap<String, HashMap<String, BufferBlock>>,
    pub descriptor_sets: HashMap<String, vk::DescriptorSet>
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
    pub flattened: Vec<GraphAction>,
    device: Rc<ash::Device>,
    pub frames: Vec<PipelineGraphFrame>,
    pub width: u32,
    pub height: u32,
    pub pipelines: HashMap<String, Rc<RefCell<Pipeline>>>,
    images: HashMap<String, Image>, // Top-level images that shouldn't be per-frame
    _sampler: Sampler, // Stored here so it doesn't get dropped
    descriptor_pool: vk::DescriptorPool
}

struct PipelineGraphFrameInfo<'a> {
    pub pipelines: &'a HashMap<String, Rc<RefCell<Pipeline>>>,
    pub descriptor_pool: vk::DescriptorPool,
    pub global_images: &'a HashMap<String, Image>,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub sampler: &'a Sampler
}

pub struct BufferBlock {
    pub size: u32,
    pub offset:u32,
    pub block_type: ReflectTypeFlags,
    pub buffer: Rc<Buffer>
}

impl PipelineGraphFrame {
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

        for (_, pipeline) in frame_info.pipelines {
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

        for (name, pipeline) in frame_info.pipelines {
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
                Vec::with_capacity(info.shader.bindings.buffers.len());

            let mut descriptor_writes: Vec<vk::WriteDescriptorSet> =
                Vec::with_capacity(info.input_images.len()  + info.output_images .len() +
                                   info.shader.bindings.buffers.len());

            {
            // Create descriptor writes and create images as needed
            let mut load_image_descriptors = |image_infos: &Vec<(String, ReflectDescriptorBinding)>| {
                for (image_name, binding) in image_infos {
                    // We only want one FILE_INPUT input image across frames as it will never change
                    if image_name == FILE_INPUT {
                        let image = &frame_info.global_images.get(FILE_INPUT).unwrap();
                        descriptor_writes.push(Self::image_write(&image, &mut desc_image_infos, binding, descriptor_set, frame_info.sampler));
                    } else {
                        match images.get(image_name) {
                            Some(image) => {
                                descriptor_writes.push(Self::image_write(&image, &mut desc_image_infos, binding, descriptor_set, frame_info.sampler));
                            }
                            None => {
                                let image = vkutils::create_image(core, image_name.to_string(), format, frame_info.width, frame_info.height);
                                descriptor_writes.push(Self::image_write(&image, &mut desc_image_infos, binding, descriptor_set, frame_info.sampler));
                                images.insert(image_name.to_string(), image);
                            }
                        }
                    }
                }
            };

            load_image_descriptors(&info.input_images);
            load_image_descriptors(&info.output_images);
            }

            {
            // Create descriptor writes and create ssbo buffers as needed
            let mut load_buffer_descriptors = |buffer_infos: &Vec<(String, ReflectDescriptorBinding)>| {
                for (buffer_name, binding) in buffer_infos {
                    let binding_idx = binding.binding;

                    match buffers.get(buffer_name) {
                        Some(buffer) => {
                            descriptor_writes.push(Self::buffer_write(&buffer, vk::DescriptorType::STORAGE_BUFFER, &mut desc_buffer_infos, binding_idx, descriptor_set));
                        }
                        None => {
                            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
                            let size = *ssbo_sizes.get(buffer_name).unwrap();
                            let buffer = vkutils::create_buffer(core, buffer_name.to_string(), size as u64, usage, gpu_allocator::MemoryLocation::CpuToGpu);
                            descriptor_writes.push(Self::buffer_write(&buffer, vk::DescriptorType::STORAGE_BUFFER, &mut desc_buffer_infos, binding_idx, descriptor_set));
                            buffers.insert(buffer_name.to_string(), buffer);
                        }
                    }
                }
            };

            load_buffer_descriptors(&info.input_ssbos);
            load_buffer_descriptors(&info.output_ssbos);

            }

            // ubos for this pipeline
            let mut pipeline_ubos: HashMap<String, BufferBlock> = HashMap::new();

            fn recurse_block(reflect_block: &ReflectBlockVariable, base_name: &String, buffer: &Rc<Buffer>, ubos: &mut HashMap<String, BufferBlock>) { // -> BufferBlock {
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

            for buffer_reflect in &info.shader.bindings.buffers {
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

        PipelineGraphFrame {
            images, buffers, ubos, descriptor_sets
        }
    }
}

impl PipelineGraph {
    pub fn flatten_graph(pipelines: &HashMap<String, Rc<RefCell<Pipeline>>>) -> Option<Vec<GraphAction>> {
        let mut flattened_graph: Vec<GraphAction> = Vec::new();
        let mut unexecuted_nodes_set: HashSet<String> = pipelines.keys().cloned().collect();

        // Function to search other other for what pipelines output into the given pipeline
        let get_input_node_names = |pipeline: &Pipeline| -> Vec<String> {
            let mut nodes: Vec<String> = Vec::new();

            let input_images: Vec<String> = pipeline.info.input_images.iter().map(|name| name.0.clone()).collect();

            for (candidate_name, candidate_pipeline) in pipelines {
                // Check if any of the outputs are used as inputs for the given pipeline
                let output_has_searched_input = candidate_pipeline.borrow().info.output_images.iter().any(|name| input_images.contains(&name.0));

                if output_has_searched_input {
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

            for node in &unexecuted_nodes {
                let pipeline = pipelines.get(node).unwrap();
                let input_nodes = get_input_node_names(&pipeline.as_ref().borrow());

                // If all input dependencies have been executed, this node is ready to execute
                let ready_to_execute = !input_nodes.iter().any(|n| unexecuted_nodes_set.contains(n));

                if ready_to_execute {
                    unexecuted_nodes_set.remove(node);
                    flattened_graph.push(GraphAction::Pipeline(Rc::clone(pipeline)));
                }
            }

            if unexecuted_nodes.len() == unexecuted_nodes_set.len() {
                warnln!("Graph incorrectly constructed. Failed to add nodes into execution: {:?}", unexecuted_nodes);
                return None
            }

            // We want barriers between groups of nodes but no barriers at the very end
            if !unexecuted_nodes_set.is_empty() {
                flattened_graph.push(GraphAction::Barrier);
            }
        }

        Some(flattened_graph)
    }

    pub unsafe fn rebuild_pipeline(&mut self, name: &str) {
        let device = &self.device;
        let mut pipeline = self.pipelines.get_mut(name).unwrap().borrow_mut();

        if let Some(shader) = Shader::new(&device, &pipeline.info.shader.path) {
            let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");

            let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo {
                module: shader.module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            };
            let pipeline_info = vk::ComputePipelineCreateInfo::builder()
                .layout(pipeline.layout.vk)
                .stage(shader_stage_create_infos);

            // In some cases, the spirv code compiles but we fail to create the pipeline due to
            // other issues, which can still be related to the shader code being wrong or
            // incompatible with current configurations
            match device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None) {
                Ok(vk_pipeline) =>  {
                    destroy_pipeline(&mut *pipeline, false);
                    pipeline.vk_pipeline = vk_pipeline[0];
                    pipeline.info.shader.module = shader.module;
                },
                Err(error) => {
                    eprintln!("{:?}", error);
                }
            }
        }
    }

    pub fn get_input_image(&self) -> &Image {
        self.images.get(FILE_INPUT).unwrap()
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
        let layout_bindings = vkutils::create_descriptor_layout_bindings(&info.shader.bindings, num_frames, pool_sizes);

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

    pub unsafe fn build_global_pipeline_data(device: Rc<ash::Device>,
                                             name: String,
                                             info: PipelineInfo,
                                             pipeline_layout: PipelineLayout) -> Rc<RefCell<Pipeline>> {


        let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");
        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo {
            module: info.shader.module,
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        };

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .layout(pipeline_layout.vk)
            .stage(shader_stage_create_infos);

        let compute_pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None).unwrap()[0];

        Rc::new(RefCell::new(Pipeline {
            device: Rc::clone(&device),
            name: name,
            info: info,
            layout: pipeline_layout,
            vk_pipeline: compute_pipeline
        }))
    }

    pub unsafe fn new(core: &VkCore, gi: PipelineGraphInfo) -> Option<PipelineGraph> {
        // Track descriptor pool sizes by descriptor type
        let mut pool_sizes: HashMap<vk::DescriptorType, u32> = HashMap::new();
        let mut pipelines: HashMap<String, Rc<RefCell<Pipeline>>> = HashMap::new();
        let mut global_images: HashMap<String, Image> = HashMap::new();

        for (name, info) in gi.pipeline_infos {
            // Build any required global images
            for (image_name, _) in &info.input_images {
                // We only want one FILE_INPUT input image across frames as it will never change
                if image_name == FILE_INPUT {
                    global_images.entry(image_name.clone()).or_insert(
                        vkutils::create_image(core, name.to_string(), gi.format, gi.width, gi.height));
                }
            }

            let pipeline_layout = Self::build_pipeline_layout(Rc::clone(&core.device), &info, &mut pool_sizes, gi.num_frames);
            let pipeline = Self::build_global_pipeline_data(Rc::clone(&core.device), name.clone(), info, pipeline_layout);
            pipelines.insert(name.to_string(), pipeline);
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
                pipelines: &pipelines,
                descriptor_pool: descriptor_pool,
                global_images: &global_images,
                width: gi.width,
                height: gi.height,
                format: gi.format,
                sampler: &sampler
            };

            frames.push(PipelineGraphFrame::new(core, &graph_frame_info));
        }

        let flattened_execution = Self::flatten_graph(&pipelines)?;

        Some(PipelineGraph {
            device: Rc::clone(&core.device),
            frames: frames,
            width: gi.width,
            height: gi.height,
            pipelines: pipelines,
            images: global_images,
            _sampler: sampler,
            descriptor_pool: descriptor_pool,
            flattened: flattened_execution
        })
    }
}

fn destroy_pipeline(pipeline: &mut Pipeline, destroy_layouts: bool) {
    let device = &pipeline.device;

    unsafe {
    device.device_wait_idle().unwrap();
    device.destroy_pipeline(pipeline.vk_pipeline, None);
    if destroy_layouts  {
        device.destroy_pipeline_layout(pipeline.layout.vk, None);
        device.destroy_descriptor_set_layout(pipeline.layout.descriptor_layout, None);
    }
    device.destroy_shader_module(pipeline.info.shader.module, None);
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        destroy_pipeline(self, true);
    }
}

impl Drop for PipelineGraph {
    fn drop(&mut self) {
        let device = &self.device;
        unsafe {

        self.device.device_wait_idle().unwrap();
        self.frames.clear();
        self.flattened.clear();
        self.pipelines.clear();

        device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

