extern crate ash;
extern crate shaderc;
extern crate gpu_allocator;
extern crate ffmpeg_next as ffmpeg;
extern crate clap;

use clap::Parser;
use gpu_allocator as gpu_alloc;
use gpu_allocator::vulkan as gpu_alloc_vk;

use ash::{vk::{self, SurfaceFormatKHR, ImageCreateFlags}};
use std::ffi::CStr;
use ash::extensions::khr;
use std::borrow::Cow;
use std::default::Default;
use std::os::raw::c_char;
use winit::window::Window;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

const NUM_FRAMES: u8 = 2;
const SHADER_PATH: &str = "shaders/shader.comp";

#[cfg(any(target_os = "macos", target_os = "ios"))]
use ash::vk::{
    KhrGetPhysicalDeviceProperties2Fn, KhrPortabilityEnumerationFn, KhrPortabilitySubsetFn,
};

#[derive(clap::Parser, Debug)]
pub struct Args {
    #[arg(short = 'i', long = "input-file")]
    file_path: String,

    #[arg(long)]
    width: Option<u32>,

    #[arg(long)]
    height: Option<u32>
}

pub struct SwapChain {
    pub surface_format: SurfaceFormatKHR,
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

pub struct ImageFileDecoder {
    decoder: ffmpeg::decoder::Video,
    input_context: ffmpeg::format::context::Input,
    width: u32,
    height: u32
}

impl ImageFileDecoder {
    pub fn new(file_path: &str) -> Self {
        let input_context = ffmpeg::format::input(&file_path).unwrap();

        let stream = input_context
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound).unwrap();

        let context_decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters()).unwrap();
        let decoder = context_decoder.decoder().video().unwrap();

        let width = decoder.width();
        let height = decoder.height();

        ImageFileDecoder {
            decoder: decoder,
            input_context: input_context,
            width: width,
            height: height
        }
    }

    pub fn decode(&mut self, buffer: *mut u8, scale_width: u32, scale_height: u32) {
        let mut scaler = ffmpeg::software::scaling::context::Context::get(
            self.decoder.format(),
            self.decoder.width(),
            self.decoder.height(),
            ffmpeg::format::Pixel::RGBA, // 8 bits per channel
            scale_width,
            scale_height,
            ffmpeg::software::scaling::flag::Flags::LANCZOS,
        ).unwrap();

        let mut frame_index = 0;

        let best_video_stream_index = self.input_context
            .streams()
            .best(ffmpeg::media::Type::Video)
            .map(|stream| stream.index());

        let mut receive_and_process_decoded_frames =
            |decoder: &mut ffmpeg::decoder::Video| -> Result<(), ffmpeg::Error> {
                let mut decoded = ffmpeg::util::frame::Video::empty();
                while decoder.receive_frame(&mut decoded).is_ok() {
                    let mut rgb_frame = ffmpeg::util::frame::Video::empty();
                    scaler.run(&decoded, &mut rgb_frame)?;
                    Self::write_frame_to_buffer(&rgb_frame, buffer);
                    frame_index += 1;

                    // TODO: Check how to the number of frames in advance
                    break;
                }
                Ok(())
            };

        for (stream, packet) in self.input_context.packets() {
            if stream.index() == best_video_stream_index.unwrap() {
                self.decoder.send_packet(&packet).unwrap();
                receive_and_process_decoded_frames(&mut self.decoder).unwrap();
            }
        }
        self.decoder.send_eof().unwrap();
        receive_and_process_decoded_frames(&mut self.decoder).unwrap();

        println!("format: {:?}", self.decoder.format());
        println!("codec format: {:?}", self.decoder.codec().unwrap().name());
    }

    fn write_frame_to_buffer(frame: &ffmpeg::util::frame::video::Video, buffer: *mut u8) {
        // https://github.com/zmwangx/rust-ffmpeg/issues/64
        let data = frame.data(0);
        let stride = frame.stride(0);
        let byte_width: usize = 4 * frame.width() as usize;
        let width: usize = frame.width() as usize;
        let height: usize = frame.height() as usize;

        let mut buffer_index: usize = 0;

        unsafe {
        // *4 for rgba
        let buffer_slice = std::slice::from_raw_parts_mut(buffer, width*height*4);

        for line in 0..height {
            let begin = line * stride;
            let end = begin + byte_width;
            let len = end-begin;
            buffer_slice[buffer_index..buffer_index+len].copy_from_slice(&data[begin..end]);
            buffer_index = buffer_index+len;
        }
        }
    }

}

use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::WindowBuilder,
};

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity, message_type, message_id_name, message_id_number, message,
    );

    vk::FALSE
}

pub struct VkSwapRes {
    pub descriptor_set: ash::vk::DescriptorSet,
}

pub struct VkFrameRes {
    pub fence: vk::Fence,
    pub present_complete_semaphore: vk::Semaphore,
    pub render_complete_semaphore: vk::Semaphore,
    pub cmd_pool: vk::CommandPool,
    pub cmd_buffer: vk::CommandBuffer,
}

pub struct VkRes {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub swapchain: SwapChain,
    pub frames: Vec<VkFrameRes>,
    pub swap_res: Vec<VkSwapRes>,
    pub queue: vk::Queue,
    pub debug_utils_loader: ash::extensions::ext::DebugUtils,
    pub debug_callback: ash::vk::DebugUtilsMessengerEXT,
    pub compute_pipeline: ash::vk::Pipeline,
    pub pipeline_layout: ash::vk::PipelineLayout,
    pub descriptor_pool: ash::vk::DescriptorPool,
    pub descriptor_layout: ash::vk::DescriptorSetLayout,
    pub input_image: Image,
    pub allocator: gpu_alloc_vk::Allocator
}

impl VkRes {
    unsafe fn create_debug_utils(instance: &ash::Instance, entry: &ash::Entry) -> (vk::DebugUtilsMessengerEXT, ash::extensions::ext::DebugUtils) {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let debug_utils_loader = ash::extensions::ext::DebugUtils::new(&entry, &instance);

        let debug_call_back = debug_utils_loader
            .create_debug_utils_messenger(&debug_info, None)
            .unwrap();

        (debug_call_back, debug_utils_loader)
    }

    unsafe fn create_instance(entry: &ash::Entry, extension_names : &Vec<*const i8>) -> ash::Instance {
        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios"))
        {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        }
        else
        {
            vk::InstanceCreateFlags::default()
        };

        let app_name = CStr::from_bytes_with_nul_unchecked(b"Reforge\0");

        let layer_names = [CStr::from_bytes_with_nul_unchecked(
            b"VK_LAYER_KHRONOS_validation\0",
        )];

        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let appinfo = vk::ApplicationInfo::builder()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&appinfo)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&extension_names)
            .flags(create_flags);

        let instance = entry
            .create_instance(&create_info, None)
            .expect("Instance creation error");


        instance
    }

    unsafe fn create_surface(entry: &ash::Entry, instance: &ash::Instance, window: &Window) -> (vk::SurfaceKHR, khr::Surface) {
        let surface_loader = khr::Surface::new(&entry, &instance);

        let surface = ash_window::create_surface(
            &entry,
            &instance,
            window.raw_display_handle(),
            window.raw_window_handle(),
            None,
        )
        .unwrap();

        return (surface, surface_loader)
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

    unsafe fn create_physical_device(instance: &ash::Instance, surface: vk::SurfaceKHR, surface_loader: &khr::Surface) -> (vk::PhysicalDevice, u32) {
        let pdevices = instance
            .enumerate_physical_devices()
            .expect("Physical device error");

        let (pdevice, queue_family_index) = pdevices
            .iter()
            .find_map(|pdevice| {
                instance
                    .get_physical_device_queue_family_properties(*pdevice)
                    .iter()
                    .enumerate()
                    .find_map(|(index, info)| {
                        let supports_graphic_and_surface =
                            info.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                                && surface_loader
                                    .get_physical_device_surface_support(
                                        *pdevice,
                                        index as u32,
                                        surface,
                                    )
                                    .unwrap();
                        if supports_graphic_and_surface {
                            Some((*pdevice, index))
                        } else {
                            None
                        }
                    })
            })
            .expect("Couldn't find suitable device.");

        return (pdevice, queue_family_index as u32)
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

        let buffer = self.device.create_buffer(&info, None).unwrap();

        let allocation = self.allocator
            .allocate(&gpu_alloc_vk::AllocationCreateDesc {
                requirements: self.device.get_buffer_memory_requirements(buffer),
                location: mem_type,
                linear: true,
                allocation_scheme: gpu_alloc_vk::AllocationScheme::GpuAllocatorManaged,
                name: "input-image-staging-buffer",
            })
            .unwrap();

        self.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()).unwrap();

        Buffer{vk: buffer, allocation: allocation}
    }

    pub unsafe fn create_input_image(device: &ash::Device, width: u32, height: u32, allocator: &mut gpu_alloc_vk::Allocator) -> Image {
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

        let image = device.create_image(&input_image_info, None).unwrap();

        let image_allocation = allocator
            .allocate(&gpu_alloc_vk::AllocationCreateDesc {
                requirements: device.get_image_memory_requirements(image),
                location: gpu_alloc::MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: gpu_alloc_vk::AllocationScheme::GpuAllocatorManaged,
                name: "input-image",
            })
            .unwrap();


        device.bind_image_memory(image, image_allocation.memory(), image_allocation.offset()).unwrap();

        let image_view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(*vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1));

        let image_view = device.create_image_view(&image_view_info, None).unwrap();

        Image {
            vk: image,
            view: image_view,
            allocation: image_allocation
        }
    }

    pub unsafe fn rebuild_changed_compute_pipeline(&mut self) {
        let shader_module = Self::create_shader_module(&self.device, SHADER_PATH);

        if shader_module.is_some() {
            let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");

            let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo {
                module: shader_module.unwrap(),
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            };
            let pipeline_info = vk::ComputePipelineCreateInfo::builder()
                .layout(self.pipeline_layout)
                .stage(shader_stage_create_infos);

            let compute_pipeline = self.device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None).unwrap()[0];

            self.compute_pipeline = compute_pipeline;
        }
    }

    unsafe fn create_swapchain(instance: &ash::Instance, device: &ash::Device, pdevice: vk::PhysicalDevice, surface: vk::SurfaceKHR, surface_loader: &khr::Surface, width: u32, height: u32) -> SwapChain {
        let surface_capabilities = surface_loader
            .get_physical_device_surface_capabilities(pdevice, surface)
            .unwrap();

        let surface_format = surface_loader
            .get_physical_device_surface_formats(pdevice, surface)
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
        let present_modes = surface_loader
            .get_physical_device_surface_present_modes(pdevice, surface)
            .unwrap();
        let present_mode = present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let swapchain_loader = khr::Swapchain::new(&instance, &device);

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
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
                device.create_image_view(&create_view_info, None).unwrap()
           }).collect();

        return SwapChain {
            surface_format: surface_format,
            vk: swapchain,
            loader: swapchain_loader,
            images: present_images,
            views: present_image_views
        };
    }

    unsafe fn create_resizable_res(instance: &ash::Instance,
                                   device: &ash::Device,
                                   pdevice: vk::PhysicalDevice,
                                   surface: vk::SurfaceKHR,
                                   surface_loader: &khr::Surface,
                                   width: u32,
                                   height: u32,
                                   input_image: &Image,
                                   descriptor_layout: &[vk::DescriptorSetLayout]) -> (SwapChain, Vec<VkSwapRes>, vk::DescriptorPool)
    {
        let swapchain = Self::create_swapchain(&instance, &device, pdevice, surface, &surface_loader, width, height);

        let descriptor_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                // Input + output, so 2*
                descriptor_count: 2*swapchain.images.len() as u32,
            },
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&descriptor_sizes)
            .max_sets(swapchain.images.len() as u32);

        let descriptor_pool = device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap();

        let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(descriptor_layout);

        let swap_res : Vec<VkSwapRes> = (0..swapchain.images.len()).map(|i|{
            let descriptor_set = device
                .allocate_descriptor_sets(&desc_alloc_info)
                .unwrap()[0];

            let input_image_descriptor = vk::DescriptorImageInfo {
                image_layout: vk::ImageLayout::GENERAL,
                image_view: input_image.view,
                ..Default::default()
            };

            let output_image_descriptor = vk::DescriptorImageInfo {
                image_layout: vk::ImageLayout::GENERAL,
                image_view: swapchain.views[i],
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
            device.update_descriptor_sets(&write_desc_sets, &[]);

            VkSwapRes {
                descriptor_set : descriptor_set
            }
        }).collect();

        (swapchain, swap_res, descriptor_pool)
    }

    unsafe fn new(window: &Window) -> Self {

        let mut extension_names = ash_window::enumerate_required_extensions(window.raw_display_handle())
                    .unwrap()
                    .to_vec();
        extension_names.push(ash::extensions::ext::DebugUtils::name().as_ptr());

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extension_names.push(KhrPortabilityEnumerationFn::name().as_ptr());
            // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
            extension_names.push(KhrGetPhysicalDeviceProperties2Fn::name().as_ptr());
        }


        let entry = ash::Entry::load().unwrap();
        let instance = Self::create_instance(&entry, &extension_names);
        let (debug_callback, debug_utils) = Self::create_debug_utils(&instance, &entry);
        let (surface, surface_loader) = Self::create_surface(&entry, &instance, &window);

        let (pdevice, queue_family_index) = Self::create_physical_device(&instance, surface, &surface_loader);

        let device_extension_names_raw : [*const i8; 1] = [
            khr::Swapchain::name().as_ptr(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            KhrPortabilitySubsetFn::name().as_ptr(),
        ];

        let features = vk::PhysicalDeviceFeatures {
            shader_clip_distance: 1,
            ..Default::default()
        };
        let priorities = [1.0];

        let queue_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&features);

        let device: ash::Device = instance
            .create_device(pdevice, &device_create_info, None)
            .unwrap();


        let frames : Vec<VkFrameRes> = (0..NUM_FRAMES).map(|_|{
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let render_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();

            let fence_create_info =
                vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

            let fence = device
                .create_fence(&fence_create_info, None)
                .expect("Create fence failed.");

            let (cmd_pool, cmd_buff) = Self::create_commands(&device, queue_family_index);

            VkFrameRes{
                fence: fence,
                present_complete_semaphore: present_complete_semaphore,
                render_complete_semaphore: render_complete_semaphore,
                cmd_pool: cmd_pool,
                cmd_buffer: cmd_buff
            }
        }).collect();

        let shader_module = Self::create_shader_module(&device, SHADER_PATH);

        let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");

        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo {
            module: shader_module.unwrap(),
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        };

        let queue = device.get_device_queue(queue_family_index, 0);

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

        let descriptor_layout = [device
            .create_descriptor_set_layout(&descriptor_info, None)
            .unwrap()];

        let window_size = window.inner_size();

        // Setting up the allocator
        let mut allocator = gpu_alloc_vk::Allocator::new(&gpu_alloc_vk::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: pdevice,
            debug_settings: Default::default(),
            buffer_device_address: false,
        }).unwrap();


        let input_image = Self::create_input_image(&device, window_size.width, window_size.height, &mut allocator);

        let (swapchain, swap_res, descriptor_pool) = Self::create_resizable_res(&instance, &device, pdevice, surface, &surface_loader, window_size.width, window_size.height, &input_image, &descriptor_layout);

        let pipeline_layout = device.
            create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&descriptor_layout), None).unwrap();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .layout(pipeline_layout)
            .stage(shader_stage_create_infos);

        let compute_pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None).unwrap()[0];


        /*
        allocator.free(allocation).unwrap();
        allocator.free(image_allocation).unwrap();
        device.destroy_buffer(test_buffer, None);
        device.destroy_image(test_image, None);
        */
    
        VkRes {
            entry: entry,
            instance: instance,
            device: device,
            swapchain: swapchain,
            frames: frames,
            swap_res: swap_res,
            queue: queue,
            debug_utils_loader: debug_utils,
            debug_callback: debug_callback,
            compute_pipeline: compute_pipeline,
            pipeline_layout: pipeline_layout,
            descriptor_pool: descriptor_pool,
            descriptor_layout: descriptor_layout[0],
            input_image: input_image,
            allocator: allocator
        }
    }
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

fn main() {
    let args = Args::parse();

    ffmpeg::init().unwrap();
    ffmpeg::log::set_level(ffmpeg::log::Level::Verbose);

    let mut file_decoder = ImageFileDecoder::new(&args.file_path);

    let window_width  = args.width.or(Some(file_decoder.width)).unwrap();
    let window_height = args.height.or(Some(file_decoder.height)).unwrap();

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
    let mut res = VkRes::new(&window);

    let mut last_modified_shader_time = get_modified_time(SHADER_PATH);

    let buffer_size = (window_width as vk::DeviceSize)*(window_height as vk::DeviceSize)*4;
    let input_image_buffer = res.create_buffer(buffer_size, vk::BufferUsageFlags::TRANSFER_SRC, gpu_alloc::MemoryLocation::CpuToGpu);

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
        let device = &res.device;

        let (present_index, _) = res
            .swapchain.loader.acquire_next_image(
                res.swapchain.vk,
                std::u64::MAX,
                frame.present_complete_semaphore, // Semaphore to signal
                vk::Fence::null(),
            )
            .unwrap();

        let swap_res = &res.swap_res[present_index as usize];

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

            let image_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image: res.input_image.vk,
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

            device.cmd_copy_buffer_to_image(frame.cmd_buffer, input_image_buffer.vk, res.input_image.vk, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[regions]);

            let image_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::GENERAL,
                image: res.input_image.vk,
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
                                    vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &[image_barrier]);

        let dispatch_x = (window_width as f32/16.0).ceil() as u32;
        let dispatch_y = (window_height as f32/16.0).ceil() as u32;

        device.cmd_bind_descriptor_sets(
            frame.cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            res.pipeline_layout,
            0,
            &[swap_res.descriptor_set],
            &[],
        );
        device.cmd_bind_pipeline(
            frame.cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            res.compute_pipeline,
        );
        device.cmd_dispatch(frame.cmd_buffer, dispatch_x, dispatch_y, 1);

        let image_barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
            old_layout: vk::ImageLayout::GENERAL,
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
                                    vk::PipelineStageFlags::COMPUTE_SHADER,
                                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::DependencyFlags::empty(), &[], &[], &[image_barrier]);


        device.end_command_buffer(frame.cmd_buffer);


        let present_complete_semaphore = &[frame.present_complete_semaphore];
        let cmd_buffers = &[frame.cmd_buffer];
        let signal_semaphores = &[frame.render_complete_semaphore];

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(present_complete_semaphore)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COMPUTE_SHADER])
            .command_buffers(cmd_buffers)
            .signal_semaphores(signal_semaphores);

        device.queue_submit(
            res.queue,
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
            .queue_present(res.queue, &present_info)
            .unwrap();

        FRAME_INDEX = (FRAME_INDEX+1)%NUM_FRAMES;
    });

    }
}
