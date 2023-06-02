extern crate ash;
extern crate shaderc;
extern crate gpu_allocator;

use gpu_allocator::vulkan as gpu_alloc_vk;

use ash::vk;
use std::ffi::CStr;
use std::default::Default;
use winit::window::Window;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt;

use crate::vulkan::core::VkCore;
use crate::vulkan::utils;
use crate::vulkan::utils::Image;

pub const NUM_FRAMES: usize = 2;

pub struct PipelineLayout {
    pub vk: vk::PipelineLayout,
    pub descriptor_layout: vk::DescriptorSetLayout
}

#[derive(Default)]
pub struct PipelineInfo {
    pub shader_path: String,
    pub input_images: Vec<(u32, String)>,
    pub output_images: Vec<(u32, String)>
}

pub struct Pipeline {
    shader_path: String,
    shader_module: vk::ShaderModule,
    pub layout: PipelineLayout,
    pub vk_pipeline: ash::vk::Pipeline
}

pub struct PipelineNode {
    pub pipeline: Rc<RefCell<Pipeline>>,
    pub name    : String,
    pub outputs : Vec<Rc<PipelineNode>>
}

pub struct PipelineGraphFrame {
    pub images: HashMap<String, Image>,
    pub descriptor_sets: HashMap<String, vk::DescriptorSet>
}

pub struct PipelineGraph {
    pub roots: Vec<Rc<PipelineNode>>,
    core: Rc<VkCore>,
    pub frames: Vec<PipelineGraphFrame>,
    pub width: u32,
    pub height: u32,
    pipelines: HashMap<String, Rc<RefCell<Pipeline>>>,
    descriptor_pool: vk::DescriptorPool
}

impl PipelineNode {
    pub fn new(name     : String,
               pipeline : Rc<RefCell<Pipeline>>,
               info     : &PipelineInfo,
               pipelines: &HashMap<String, Rc<RefCell<Pipeline>>>,
               infos    : &HashMap<&str, PipelineInfo>) -> PipelineNode {

        let mut outputs: Vec<Rc<PipelineNode>> = Vec::new();

        for (_, output_name) in &info.output_images {
            let matching_input_pipelines = PipelineGraph::get_pipelines_with_input(&output_name, &pipelines, &infos);

            for (name, info, matching_pipeline) in &matching_input_pipelines {
                outputs.push(Rc::new(PipelineNode::new(name.to_string(), Rc::clone(matching_pipeline), info, pipelines, infos)));
            }
        }

        PipelineNode { pipeline, name, outputs }
    }
}

impl fmt::Debug for PipelineGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn traverse(node: &PipelineNode, f: &mut fmt::Formatter<'_>) { 
            f.write_fmt(format_args!("{}", node.name)).unwrap();

            for output in &node.outputs {
                f.write_str(" -> ").unwrap();
                traverse(&output, f);
            }
        }

        for node in &self.roots {
            traverse(node, f);

            if self.roots.len() > 1 {
                f.write_str("\n").unwrap();
            }
        }

        Ok(())
    }
}

struct PipelineGraphFrameInfo<'a> {
    pub pipelines: &'a HashMap<String, Rc<RefCell<Pipeline>>>,
    pub pipeline_infos: &'a HashMap<&'a str, PipelineInfo>,
    pub descriptor_pool: vk::DescriptorPool,
    pub num_frames: usize,
    pub width: u32,
    pub height: u32
}

impl PipelineGraphFrame {
    unsafe fn storage_image_write(image: &Image, infos: &mut Vec<vk::DescriptorImageInfo>, desc_idx: u32, set: vk::DescriptorSet) -> vk::WriteDescriptorSet {

        infos.push(vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::GENERAL,
            image_view: image.view,
            ..Default::default()
        });

        vk::WriteDescriptorSet {
            dst_set: set,
            dst_binding: desc_idx,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            p_image_info: infos.last().unwrap(),
            ..Default::default()
        }
    }

    unsafe fn new_vec(device: &ash::Device, allocator: &mut gpu_alloc_vk::Allocator, frame_info: &PipelineGraphFrameInfo) -> Vec<PipelineGraphFrame> {
        let mut frames: Vec<PipelineGraphFrame> = Vec::with_capacity(frame_info.num_frames);

        if frame_info.pipeline_infos.len() == 0  {
            return frames;
        }

        // Create per-frame images, descriptor sets
        for i in 0..frame_info.num_frames {
            let mut images: HashMap<String, Image> = HashMap::new();
            let mut descriptor_sets: HashMap<String, vk::DescriptorSet> = HashMap::new();

            for (pipeline_name, info) in frame_info.pipeline_infos {
                let pipeline = &frame_info.pipelines.get(&pipeline_name.to_string()).unwrap();

                let layout_info = &[pipeline.borrow().layout.descriptor_layout];
                let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(frame_info.descriptor_pool)
                    .set_layouts(layout_info);

                let descriptor_set = device
                    .allocate_descriptor_sets(&desc_alloc_info)
                    .unwrap()[0];

                let mut desc_image_infos: Vec<vk::DescriptorImageInfo> =
                    Vec::with_capacity(info.input_images.len() + info.output_images.len());
                let mut descriptor_writes: Vec<vk::WriteDescriptorSet> =
                    Vec::with_capacity(info.input_images.len() + info.output_images.len());

                // Input images
                for (desc_idx, image_name) in &info.input_images {
                    // We only want one "file" input image across frames as it will never change
                    if i > 0 && image_name == "file" {
                        let image = &frames[0].images.get("file").unwrap();
                        descriptor_writes.push(Self::storage_image_write(&image, &mut desc_image_infos, *desc_idx, descriptor_set));
                    } else {
                        match images.get(image_name) {
                            Some(image) => {
                                descriptor_writes.push(Self::storage_image_write(&image, &mut desc_image_infos, *desc_idx, descriptor_set));
                            }
                            None => {
                                let image = utils::create_image(device, image_name.to_string(), frame_info.width, frame_info.height, allocator);
                                descriptor_writes.push(Self::storage_image_write(&image, &mut desc_image_infos, *desc_idx, descriptor_set));
                                images.insert(image_name.to_string(), image);
                            }
                        }
                    }
                }

                // Output images
                for (desc_idx, image_name) in &info.output_images {
                    match images.get(image_name) {
                        Some(image) => {
                            descriptor_writes.push(Self::storage_image_write(&image, &mut desc_image_infos, *desc_idx, descriptor_set));
                        }
                        None => {
                            let image = utils::create_image(device, image_name.to_string(), frame_info.width, frame_info.height, allocator);
                            descriptor_writes.push(Self::storage_image_write(&image, &mut desc_image_infos, *desc_idx, descriptor_set));
                            images.insert(image_name.to_string(), image);
                        }
                    }
                }

                device.update_descriptor_sets(&descriptor_writes, &[]);
                descriptor_sets.insert(pipeline_name.to_string(), descriptor_set);
            }

            frames.push(PipelineGraphFrame {
                images, descriptor_sets
            });
        }

    frames

    }
}

impl PipelineGraph {
    pub fn get_pipelines_with_input<'a>(name     : &str,
                                        pipelines: &HashMap<String, Rc<RefCell<Pipeline>>>,
                                        infos    : &'a HashMap<&str, PipelineInfo>) -> Vec<(String, &'a PipelineInfo, Rc<RefCell<Pipeline>>)> {

        let mut matching_pipelines: Vec<(String, &PipelineInfo, Rc<RefCell<Pipeline>>)> = Vec::new();

        for (pipeline_name, info) in infos {
            for (_, image_name) in &info.input_images {
                if image_name == name {
                    matching_pipelines.push((pipeline_name.to_string(), &info, Rc::clone(pipelines.get(&pipeline_name.to_string()).unwrap())));
                }
            }
        }

        matching_pipelines
    }

    pub fn create_nodes(pipelines: &HashMap<String, Rc<RefCell<Pipeline>>>, pipeline_infos: &HashMap<&str, PipelineInfo>) -> Vec<Rc<PipelineNode>> {
        let mut roots: Vec<Rc<PipelineNode>> = Vec::new();
        let matching_root_pipelines = Self::get_pipelines_with_input("file", pipelines, &pipeline_infos);

        for (name, info, pipeline) in &matching_root_pipelines {
            roots.push(Rc::new(PipelineNode::new(name.to_string(), Rc::clone(pipeline), info, pipelines, pipeline_infos)));
        }

        roots
    }

    pub unsafe fn rebuild_pipeline(&mut self, name: &str) {
        let mut pipeline = self.pipelines.get_mut(name).unwrap().borrow_mut();
        let shader_module = Self::create_shader_module(&self.core.device, &pipeline.shader_path);

        if shader_module.is_some() {
            let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");

            let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo {
                module: shader_module.unwrap(),
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            };
            let pipeline_info = vk::ComputePipelineCreateInfo::builder()
                .layout(pipeline.layout.vk)
                .stage(shader_stage_create_infos);

            pipeline.vk_pipeline = self.core.device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None).unwrap()[0];
            pipeline.shader_module = shader_module.unwrap();
        }
    }

    pub fn get_input_image(&self) -> &Image {
        &self.frames[0].images.get("file").unwrap()
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

    pub unsafe fn build_global_pipeline_data(device: &ash::Device, infos: &HashMap<&str, PipelineInfo>) -> (HashMap<String, Rc<RefCell<Pipeline>>>, vk::DescriptorPool) {
        let mut pipelines: HashMap<String, Rc<RefCell<Pipeline>>> = HashMap::new();
        let mut descriptor_pool: vk::DescriptorPool = vk::DescriptorPool::null();

        if infos.len() == 0  {
            return (pipelines, descriptor_pool);
        }

        // Track descriptor pool sizes by descriptor type
        let mut descriptor_size_map : HashMap<vk::DescriptorType, u32> = HashMap::new();
        let mut found_swapchain_image = false;

        let mut image_binding = vk::DescriptorSetLayoutBinding {
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            binding: 0,
            ..Default::default()
        };

        // descriptor layouts, add descriptor pool sizes, and add pipelines to hashmap
        for (pipeline_name, info) in infos {
            let mut descriptor_layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = Vec::with_capacity(infos.len());

            // Input images
            for (desc_idx, _) in &info.input_images {
                *descriptor_size_map.entry(vk::DescriptorType::STORAGE_IMAGE).or_insert(0) += 1*NUM_FRAMES as u32;
                image_binding.binding = *desc_idx;
                descriptor_layout_bindings.push(image_binding);
            }

            // Output images
            for (desc_idx, image_name) in &info.output_images {
                if image_name == "swapchain"  {
                    found_swapchain_image = true;
                }

                *descriptor_size_map.entry(vk::DescriptorType::STORAGE_IMAGE).or_insert(0) += 1*NUM_FRAMES as u32;
                image_binding.binding = *desc_idx;
                descriptor_layout_bindings.push(image_binding);
            }

            let descriptor_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&descriptor_layout_bindings);
            let descriptor_layout = [device
                .create_descriptor_set_layout(&descriptor_info, None)
                .unwrap()];

            let pipeline_layout = device.
                create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&descriptor_layout), None).unwrap();

            let shader_module = Self::create_shader_module(&device, &info.shader_path);
            let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");
            let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo {
                module: shader_module.unwrap(),
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            };

            let pipeline_info = vk::ComputePipelineCreateInfo::builder()
                .layout(pipeline_layout)
                .stage(shader_stage_create_infos);

            let compute_pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None).unwrap()[0];

            let pipeline_layout = PipelineLayout {
                vk: pipeline_layout,
                descriptor_layout: descriptor_layout[0]
            };

            pipelines.insert(pipeline_name.to_string(), Rc::new(RefCell::new(Pipeline {
                shader_path: info.shader_path.clone(),
                shader_module : shader_module.unwrap(),
                layout: pipeline_layout,
                vk_pipeline: compute_pipeline
            })));
        }

        if !found_swapchain_image  {
            eprintln!("No output named \"swapchain\", which is currently required");
            std::process::exit(1);
        }

        // Create descriptor pool
        let descriptor_size_vec = Self::map_to_desc_size(&descriptor_size_map);

        // We determine number of sets by the number of pipelines
        // Generally, each pipeline will have NUM_FRAMES amount of descriptor copies
        // However, if there is a set of swapchain images being used for one pipeline, we will include that
        let num_max_sets = NUM_FRAMES as u32*infos.len() as u32;

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&descriptor_size_vec)
            .max_sets(num_max_sets);

        descriptor_pool = device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap();

        (pipelines, descriptor_pool)
    }

    unsafe fn create_shader_module(device: &ash::Device, path: &str) -> Option<vk::ShaderModule> {
        let glsl_source = std::fs::read_to_string(path)
            .expect("Should have been able to read the file");

        let compiler = shaderc::Compiler::new().unwrap();
        let options = shaderc::CompileOptions::new().unwrap();

        match compiler.compile_into_spirv(&glsl_source.to_owned(),
                                                        shaderc::ShaderKind::Compute,
                                                        path,
                                                        "main",
                                                        Some(&options)) {
            Ok(binary) => {
                assert_eq!(Some(&0x07230203), binary.as_binary().first());

                let shader_info = vk::ShaderModuleCreateInfo::builder().code(&binary.as_binary());

                Some(device.create_shader_module(&shader_info, None).expect("Shader module error"))
            }
            Err(e) => { eprintln!("{:?}", e); None }
        }
    }

    pub unsafe fn new(core: Rc<VkCore>, allocator: &mut gpu_alloc_vk::Allocator, pipeline_infos: &HashMap<&str, PipelineInfo>, window: &Window) -> Self {
        let (pipelines, descriptor_pool) = Self::build_global_pipeline_data(&core.device, &pipeline_infos);

        let window_size = window.inner_size();

        let graph_frame_info = PipelineGraphFrameInfo {
            pipelines: &pipelines,
            pipeline_infos: pipeline_infos,
            descriptor_pool: descriptor_pool,
            num_frames: NUM_FRAMES,
            width: window_size.width,
            height: window_size.height
        };

        /*
        allocator.free(allocation).unwrap();
        allocator.free(image_allocation).unwrap();
        device.destroy_buffer(test_buffer, none);
        device.destroy_image(test_image, none);
        */
    
        let roots = Self::create_nodes(&pipelines, pipeline_infos);
        let frames = PipelineGraphFrame::new_vec(&core.device, allocator, &graph_frame_info);

        PipelineGraph {
            core: core,
            frames: frames,
            width: window_size.width,
            height: window_size.height,
            pipelines: pipelines,
            descriptor_pool: descriptor_pool,
            roots: roots
        }
    }
}

