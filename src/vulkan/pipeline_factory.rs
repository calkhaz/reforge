extern crate ash;
extern crate shaderc;
extern crate gpu_allocator;

use gpu_allocator as gpu_alloc;
use gpu_allocator::vulkan as gpu_alloc_vk;

use ash::vk;
use std::ffi::CStr;
use ash::extensions::khr;
use std::default::Default;
use winit::window::Window;
use std::collections::HashMap;
use std::rc::Rc;

use crate::vulkan::core::VkCore;

pub const NUM_FRAMES: u8 = 2;
pub const SHADER_PATH: &str = "shaders/shader.comp";

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
}

pub struct PipelineLayout {
    shader_module: vk::ShaderModule,
    pub vk: vk::PipelineLayout,
    pub descriptor_layout: vk::DescriptorSetLayout
}

#[derive(Default)]
pub struct PipelineInfo {
    pub shader_path: String,
    pub input_images: Vec<(u32, String)>,
    pub output_images: Vec<(u32, String)>
}

pub struct PipelineFactory {
    core: Rc<VkCore>,
    pub swapchain: SwapChain,
    pub frames: Vec<VkFrameRes>,
    pub compute_pipeline: ash::vk::Pipeline,
    pub pipeline_layout: PipelineLayout,
    pub allocator: gpu_alloc_vk::Allocator,
    pub images: HashMap<String, Image>,
    pipeline_infos: HashMap<String, PipelineInfo>
}

impl PipelineFactory {

    /*(
    pub unsafe fn add(&mut self, name: &str) -> &mut PipelineInfo {
        //self.pipeline_infos.insert(name, PipelineInfo());
        let info : PipelineInfo = Default::default();
        self.pipeline_infos.entry(name.to_string()).or_insert_with(|| info)
    }
    */

    pub unsafe fn add(&mut self, name: &str, info: PipelineInfo)  {
        self.pipeline_infos.insert(name.to_string(), info);
    }

    pub unsafe fn build(&mut self) {
        for (key, info) in &self.pipeline_infos {
            for (key, value) in &info.input_images {

            }
            for (key, value) in &info.output_images {
            }
        }
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

    pub unsafe fn create_image(&mut self, name: String, width: u32, height: u32) -> &Image {
        let input_image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .format(vk::Format::R8G8B8A8_UNORM)
            .mip_levels(1)
            .extent(vk::Extent3D{width: width, height: height, depth: 1})
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST)
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

        let image = Image {
            vk: vk_image,
            view: image_view,
            allocation: image_allocation
        };

        self.images.insert(name.clone(), image);

        &self.images.get(&name).unwrap()
    }

    pub unsafe fn get_image(&self, name: String) -> &Image {
        &self.images.get(&name).unwrap()
    }

    pub unsafe fn rebuild_changed_compute_pipeline(&mut self) {
        let shader_module = Self::create_shader_module(&self.core.device, SHADER_PATH);

        if shader_module.is_some() {
            let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");

            let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo {
                module: shader_module.unwrap(),
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            };
            let pipeline_info = vk::ComputePipelineCreateInfo::builder()
                .layout(self.pipeline_layout.vk)
                .stage(shader_stage_create_infos);

            let compute_pipeline = self.core.device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None).unwrap()[0];

            self.compute_pipeline = compute_pipeline;
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
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
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
                cmd_buffer: cmd_buff
            }
        }).collect();

        let shader_module = Self::create_shader_module(&core.device, SHADER_PATH);

        let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");

        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo {
            module: shader_module.unwrap(),
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        };

        let desc_layout_bindings = [
            // input
            vk::DescriptorSetLayoutBinding {
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                binding: 0,
                ..Default::default()
            },
            // output
            vk::DescriptorSetLayoutBinding {
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                binding: 1,
                ..Default::default()
            }
        ];
        let descriptor_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_layout_bindings);

        let descriptor_layout = [core.device
            .create_descriptor_set_layout(&descriptor_info, None)
            .unwrap()];

        let window_size = window.inner_size();

        // Setting up the allocator
        let mut allocator = gpu_alloc_vk::Allocator::new(&gpu_alloc_vk::AllocatorCreateDesc {
            instance: core.instance.clone(),
            device: core.device.clone(),
            physical_device: core.pdevice,
            debug_settings: Default::default(),
            buffer_device_address: false,
        }).unwrap();


        let swapchain = Self::create_swapchain(&core, window_size.width, window_size.height);

        let pipeline_layout = core.device.
            create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&descriptor_layout), None).unwrap();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .layout(pipeline_layout)
            .stage(shader_stage_create_infos);

        let compute_pipeline = core.device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None).unwrap()[0];

        let pipeline_layout = PipelineLayout {
            shader_module : shader_module.unwrap(),
            vk: pipeline_layout,
            descriptor_layout: descriptor_layout[0]
        };


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
            compute_pipeline: compute_pipeline,
            pipeline_layout: pipeline_layout,
            allocator: allocator,
            images: HashMap::new(),
            pipeline_infos : HashMap::new()
        }
    }
}

