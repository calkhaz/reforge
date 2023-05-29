extern crate ash;
extern crate shaderc;
extern crate gpu_allocator;

use gpu_allocator as gpu_alloc;
use gpu_allocator::vulkan as gpu_alloc_vk;

use ash::vk;
use std::ffi::CStr;
use std::hash::Hash;
use ash::extensions::khr;
use std::default::Default;
use winit::window::Window;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

use crate::vulkan::core::VkCore;

pub const NUM_FRAMES: u8 = 2;

pub struct SwapChain {
    pub surface_format: vk::SurfaceFormatKHR,
    pub vk: vk::SwapchainKHR,
    pub loader: khr::Swapchain,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>
}

pub struct Image {
    pub vk: vk::Image,
    pub view: vk::ImageView,
    pub allocation: gpu_alloc_vk::Allocation
}

pub struct Buffer {
    pub vk: vk::Buffer,
    pub allocation: gpu_alloc_vk::Allocation
}


pub struct VkFrameRes {
    pub fence: vk::Fence,
    pub present_complete_semaphore: vk::Semaphore,
    pub render_complete_semaphore: vk::Semaphore,
    pub cmd_pool: vk::CommandPool,
    pub cmd_buffer: vk::CommandBuffer,
    pub images: HashMap<String, Image>,
    pub descriptor_sets: HashMap<String, vk::DescriptorSet>
}

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

pub struct PipelineGraph {
    pub roots: Vec<Rc<PipelineNode>>
}

pub struct PipelineFactory {
    core: Rc<VkCore>,
    pub swapchain: SwapChain,
    pub frames: Vec<VkFrameRes>,
    pub allocator: gpu_alloc_vk::Allocator,
    pub pipeline_infos: HashMap<String, PipelineInfo>,
    width: u32,
    height: u32,
    pub pipelines: HashMap<String, Rc<RefCell<Pipeline>>>,
    descriptor_pool: vk::DescriptorPool
}

impl PipelineNode {
    pub fn new(name     : String,
               pipeline : Rc<RefCell<Pipeline>>,
               info     : &PipelineInfo,
               pipelines: &HashMap<String, Rc<RefCell<Pipeline>>>,
               infos    : &HashMap<String, PipelineInfo>) -> PipelineNode {

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

impl PipelineGraph {
    pub fn get_pipelines_with_input<'a>(name     : &str,
                                        pipelines: &HashMap<String, Rc<RefCell<Pipeline>>>,
                                        infos    : &'a HashMap<String, PipelineInfo>) -> Vec<(String, &'a PipelineInfo, Rc<RefCell<Pipeline>>)> {

        let mut matching_pipelines: Vec<(String, &PipelineInfo, Rc<RefCell<Pipeline>>)> = Vec::new();

        for (pipeline_name, info) in infos {
            for (_, image_name) in &info.input_images {
                if image_name == name {
                    matching_pipelines.push((pipeline_name.to_string(), &info, Rc::clone(pipelines.get(pipeline_name).unwrap())));
                }
            }
        }

        matching_pipelines
    }

    pub fn new(pipelines: &HashMap<String, Rc<RefCell<Pipeline>>>,
               infos    : &HashMap<String, PipelineInfo>) -> PipelineGraph {

        let mut roots: Vec<Rc<PipelineNode>> = Vec::new();

        let matching_root_pipelines = Self::get_pipelines_with_input("file", &pipelines, &infos);

        for (name, info, pipeline) in &matching_root_pipelines {
            roots.push(Rc::new(PipelineNode::new(name.to_string(), Rc::clone(pipeline), info, pipelines, infos)));
        }

        PipelineGraph { roots }
    }
}

impl PipelineFactory {

    /*(
    pub unsafe fn add(&mut self, name: &str) -> &mut PipelineInfo {
        //self.pipeline_infos.insert(name, PipelineInfo());
        let info : PipelineInfo = Default::default();
        self.pipeline_infos.entry(name.to_string()).or_insert_with(|| info)
    }
    */

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

    pub unsafe fn add(&mut self, name: &str, info: PipelineInfo)  {
        self.pipeline_infos.insert(name.to_string(), info);
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

    pub unsafe fn build(&mut self) {
        if self.pipeline_infos.len() == 0  {
            return;
        }

        // Move data out to not borrow &self mutably/immutably at the same time
        let infos = std::mem::replace(&mut self.pipeline_infos, HashMap::new());

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
        for (pipeline_name, info) in &infos {
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
            let descriptor_layout = [self.core.device
                .create_descriptor_set_layout(&descriptor_info, None)
                .unwrap()];

            let pipeline_layout = self.core.device.
                create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&descriptor_layout), None).unwrap();

            let shader_module = Self::create_shader_module(&self.core.device, &info.shader_path);
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

            let compute_pipeline = self.core.device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None).unwrap()[0];

            let pipeline_layout = PipelineLayout {
                vk: pipeline_layout,
                descriptor_layout: descriptor_layout[0]
            };

            self.pipelines.insert(pipeline_name.to_string(), Rc::new(RefCell::new(Pipeline {
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

        self.descriptor_pool = self.core.device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap();

        // Create per-frame images, descriptor sets
        for (pipeline_name, info) in &infos {

            let pipeline = &self.pipelines.get(pipeline_name).unwrap();

            let layout_info = &[pipeline.borrow().layout.descriptor_layout];
            let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(layout_info);

            for i in 0..self.frames.len() {
                let mut images: HashMap<String, Image> = HashMap::new();


                let descriptor_set = self.core.device
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
                        let image = &self.frames[0].images.get("file").unwrap();
                        descriptor_writes.push(Self::storage_image_write(&image, &mut desc_image_infos, *desc_idx, descriptor_set));
                    } else {
                        match self.frames[i].images.get(image_name) {
                            Some(image) => {
                                descriptor_writes.push(Self::storage_image_write(&image, &mut desc_image_infos, *desc_idx, descriptor_set));
                            }
                            None => {
                                let image = self.create_image(image_name.to_string(), self.width, self.height);
                                descriptor_writes.push(Self::storage_image_write(&image, &mut desc_image_infos, *desc_idx, descriptor_set));
                                images.insert(image_name.to_string(), image);
                            }
                        }
                    }
                }

                // Output images
                for (desc_idx, image_name) in &info.output_images {
                    match self.frames[i].images.get(image_name) {
                        Some(image) => {
                            descriptor_writes.push(Self::storage_image_write(&image, &mut desc_image_infos, *desc_idx, descriptor_set));
                        }
                        None => {
                            let image = self.create_image(image_name.to_string(), self.width, self.height);
                            descriptor_writes.push(Self::storage_image_write(&image, &mut desc_image_infos, *desc_idx, descriptor_set));
                            images.insert(image_name.to_string(), image);
                        }
                    }
                }

                self.core.device.update_descriptor_sets(&descriptor_writes, &[]);
                self.frames[i].descriptor_sets.insert(pipeline_name.to_string(), descriptor_set);
                self.frames[i].images.extend(images);
            }

        }

        let graph = PipelineGraph::new(&self.pipelines, &self.pipeline_infos);


        // Put data back
        self.pipeline_infos = infos;
    }

    unsafe fn create_commands(device: &ash::Device, queue_family_index: u32) -> (vk::CommandPool, vk::CommandBuffer) {
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);

        let pool = device.create_command_pool(&pool_create_info, None).unwrap();

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffer = device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap();

        (pool, command_buffer[0])
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

    pub unsafe fn create_buffer(&mut self,
                                size: vk::DeviceSize,
                                usage: vk::BufferUsageFlags,
                                mem_type: gpu_alloc::MemoryLocation) -> Buffer {
        let info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = self.core.device.create_buffer(&info, None).unwrap();

        let allocation = self.allocator
            .allocate(&gpu_alloc_vk::AllocationCreateDesc {
                requirements: self.core.device.get_buffer_memory_requirements(buffer),
                location: mem_type,
                linear: true,
                allocation_scheme: gpu_alloc_vk::AllocationScheme::GpuAllocatorManaged,
                name: "input-image-staging-buffer",
            })
            .unwrap();

        self.core.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()).unwrap();

        Buffer{vk: buffer, allocation: allocation}
    }

    pub unsafe fn create_image(&mut self, name: String, width: u32, height: u32) -> Image {
        let input_image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .format(vk::Format::R8G8B8A8_UNORM)
            .mip_levels(1)
            .extent(vk::Extent3D{width: width, height: height, depth: 1})
            // TOOD: Optimize for what is actually needed
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let vk_image = self.core.device.create_image(&input_image_info, None).unwrap();

        let image_allocation = self.allocator
            .allocate(&gpu_alloc_vk::AllocationCreateDesc {
                requirements: self.core.device.get_image_memory_requirements(vk_image),
                location: gpu_alloc::MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: gpu_alloc_vk::AllocationScheme::GpuAllocatorManaged,
                name: &name
            })
            .unwrap();


        self.core.device.bind_image_memory(vk_image, image_allocation.memory(), image_allocation.offset()).unwrap();

        let image_view_info = vk::ImageViewCreateInfo::builder()
            .image(vk_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(*vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1));

        let image_view = self.core.device.create_image_view(&image_view_info, None).unwrap();

        Image {
            vk: vk_image,
            view: image_view,
            allocation: image_allocation
        }
    }

    unsafe fn create_swapchain(core: &VkCore, width: u32, height: u32) -> SwapChain {
        let surface_capabilities = core.surface_loader
            .get_physical_device_surface_capabilities(core.pdevice, core.surface)
            .unwrap();

        let surface_format = core.surface_loader
            .get_physical_device_surface_formats(core.pdevice, core.surface)
            .unwrap()[0];

        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }
        let surface_resolution = match surface_capabilities.current_extent.width {
            std::u32::MAX => vk::Extent2D {
                width: width,
                height: height,
            },
            _ => surface_capabilities.current_extent,
        };
        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };
        let present_modes = core.surface_loader
            .get_physical_device_surface_present_modes(core.pdevice, core.surface)
            .unwrap();
        let present_mode = present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let swapchain_loader = khr::Swapchain::new(&core.instance, &core.device);

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(core.surface)
            .min_image_count(desired_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_resolution)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE| vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1);

        let swapchain = swapchain_loader
            .create_swapchain(&swapchain_create_info, None)
            .unwrap();

        let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();

        let present_image_views: Vec<vk::ImageView> = present_images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);
                core.device.create_image_view(&create_view_info, None).unwrap()
           }).collect();

        return SwapChain {
            surface_format: surface_format,
            vk: swapchain,
            loader: swapchain_loader,
            images: present_images,
            views: present_image_views
        };
    }

    pub unsafe fn new(core: Rc<VkCore>, window: &Window) -> Self {
        let frames : Vec<VkFrameRes> = (0..NUM_FRAMES).map(|_|{
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore = core.device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let render_complete_semaphore = core.device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            let fence_create_info =
                vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

            let fence = core.device
                .create_fence(&fence_create_info, None)
                .expect("Create fence failed.");

            let (cmd_pool, cmd_buff) = Self::create_commands(&core.device, core.queue_family_index);

            VkFrameRes{
                fence: fence,
                present_complete_semaphore: present_complete_semaphore,
                render_complete_semaphore: render_complete_semaphore,
                cmd_pool: cmd_pool,
                cmd_buffer: cmd_buff,
                images: HashMap::new(),
                descriptor_sets: HashMap::new()
            }
        }).collect();

        let window_size = window.inner_size();

        // Setting up the allocator
        let allocator = gpu_alloc_vk::Allocator::new(&gpu_alloc_vk::AllocatorCreateDesc {
            instance: core.instance.clone(),
            device: core.device.clone(),
            physical_device: core.pdevice,
            debug_settings: Default::default(),
            buffer_device_address: false,
        }).unwrap();


        let swapchain = Self::create_swapchain(&core, window_size.width, window_size.height);

        /*
        allocator.free(allocation).unwrap();
        allocator.free(image_allocation).unwrap();
        device.destroy_buffer(test_buffer, None);
        device.destroy_image(test_image, None);
        */
    
        PipelineFactory {
            core: core,
            swapchain: swapchain,
            frames: frames,
            allocator: allocator,
            pipeline_infos : HashMap::new(),
            width: window_size.width,
            height: window_size.height,
            pipelines: HashMap::new(),
            descriptor_pool: vk::DescriptorPool::null()
        }
    }
}

