extern crate ash;
use ash::{vk::{self, SurfaceFormatKHR, CommandBuffer}, Entry};
pub use ash::{Device, Instance};
use ash_window::create_surface;
use std::io::Cursor;

use std::{ffi::CStr, borrow::BorrowMut};

use ash::extensions::khr;
//use ash::extensions::{
//    ext::DebugUtils,
//    khr::{Surface, Swapchain},
//};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::borrow::Cow;
use std::cell::RefCell;
use std::default::Default;
use std::ops::Drop;
use std::os::raw::c_char;
use winit::window::Window;

const NUM_FRAMES: u8 = 2;

#[cfg(any(target_os = "macos", target_os = "ios"))]
use ash::vk::{
    KhrGetPhysicalDeviceProperties2Fn, KhrPortabilityEnumerationFn, KhrPortabilitySubsetFn,
};

pub struct SwapChain {
    pub surface_format: SurfaceFormatKHR,
    pub vk: vk::SwapchainKHR,
    pub loader: khr::Swapchain,
    pub images: Vec<vk::ImageView>
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

pub struct VkFrameRes {
    pub fence: vk::Fence,
    pub present_complete_semaphore: vk::Semaphore,
    pub render_complete_semaphore: vk::Semaphore,
    pub cmd_pool: vk::CommandPool,
    pub cmd_buffer: vk::CommandBuffer
}

pub struct VkRes {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub swapchain: SwapChain,
    pub frames: Vec<VkFrameRes>,
    pub queue: vk::Queue,
//    pub surface_loader: Surface,
//    pub swapchain_loader: Swapchain,
    pub debug_utils_loader: ash::extensions::ext::DebugUtils,
    pub debug_callback: ash::vk::DebugUtilsMessengerEXT,
    pub compute_pipeline: ash::vk::Pipeline,
    pub pipeline_layout: ash::vk::PipelineLayout,
    pub descriptor_set: ash::vk::DescriptorSet,
    pub descriptor_pool: ash::vk::DescriptorPool,
    pub descriptor_layout: ash::vk::DescriptorSetLayout,
////    pub window: winit::window::Window,
////    pub event_loop: RefCell<EventLoop<()>>,
//    pub debug_call_back: vk::DebugUtilsMessengerEXT,
//
//    pub pdevice: vk::PhysicalDevice,
// //   pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
//    pub queue_family_index: u32,
//    pub present_queue: vk::Queue,

//    pub surface: vk::SurfaceKHR,
//    pub surface_format: vk::SurfaceFormatKHR,
//    pub surface_resolution: vk::Extent2D,
//
//    pub swapchain: vk::SwapchainKHR,
//    pub present_images: Vec<vk::Image>,
//    pub present_image_views: Vec<vk::ImageView>,
//
//    pub pool: vk::CommandPool,
//    pub draw_command_buffer: vk::CommandBuffer,
//    pub setup_command_buffer: vk::CommandBuffer,
//
//    pub depth_image: vk::Image,
//    pub depth_image_view: vk::ImageView,
//    pub depth_image_memory: vk::DeviceMemory,
//
//    pub present_complete_semaphore: vk::Semaphore,
//    pub rendering_complete_semaphore: vk::Semaphore,
//
//    pub draw_commands_reuse_fence: vk::Fence,
//    pub setup_commands_reuse_fence: vk::Fence,
}

impl VkRes {
    unsafe fn create_debug_utils(instance : &Instance, entry: &Entry) -> (vk::DebugUtilsMessengerEXT, ash::extensions::ext::DebugUtils) {
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

    unsafe fn create_instance(entry: &Entry, extension_names : &Vec<*const i8>) -> Instance {
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

        let instance: Instance = entry
            .create_instance(&create_info, None)
            .expect("Instance creation error");


        instance
    }

    unsafe fn create_surface(entry: &Entry, instance: &Instance, window: &Window) -> (vk::SurfaceKHR, khr::Surface) {
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

    unsafe fn create_physical_device(instance: &Instance, surface: vk::SurfaceKHR, surface_loader: &khr::Surface) -> (vk::PhysicalDevice, u32) {
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

    unsafe fn create_swapchain(instance: &Instance, device: &Device, pdevice: vk::PhysicalDevice, surface: vk::SurfaceKHR, surface_loader: &khr::Surface, width: u32, height: u32) -> SwapChain {
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
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
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
            images: present_image_views
        };
    }

    unsafe fn new(event_loop: &EventLoop<()>, window: &Window) -> Self {

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


        let entry = Entry::load().unwrap();
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

        let device: Device = instance
            .create_device(pdevice, &device_create_info, None)
            .unwrap();

        let swapchain = Self::create_swapchain(&instance, &device, pdevice, surface, &surface_loader, 800, 600);

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

        let mut spv_file = Cursor::new(&include_bytes!("../shaders/compute.spv")[..]);

        let shader_code = ash::util::read_spv(&mut spv_file).expect("Failed to read vertex shader spv file");
        let shader_info = vk::ShaderModuleCreateInfo::builder().code(&shader_code);

        let shader_module = device
            .create_shader_module(&shader_info, None)
            .expect("Vertex shader module error");

        let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");

        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo {
                module: shader_module,
                p_name: shader_entry_name.as_ptr(),
                stage: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
        };


        let queue = device.get_device_queue(queue_family_index, 0);

        let descriptor_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
            },
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&descriptor_sizes)
            .max_sets(1);

        let descriptor_pool = device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap();
        let desc_layout_bindings = [
            vk::DescriptorSetLayoutBinding {
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            }
        ];
        let descriptor_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_layout_bindings);

        let descriptor_layout = [device
            .create_descriptor_set_layout(&descriptor_info, None)
            .unwrap()];

        let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_layout);
        let descriptor_set = device
            .allocate_descriptor_sets(&desc_alloc_info)
            .unwrap();

        let pipeline_layout = device.
            create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&descriptor_layout), None).unwrap();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .layout(pipeline_layout)
            .stage(shader_stage_create_infos);

        let compute_pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None).unwrap()[0];
    
        VkRes {
            entry: entry,
            instance: instance,
            device: device,
            swapchain: swapchain,
            frames: frames,
            queue: queue,
            debug_utils_loader: debug_utils,
            debug_callback: debug_callback,
            compute_pipeline: compute_pipeline,
            pipeline_layout: pipeline_layout,
            descriptor_set: descriptor_set[0],
            descriptor_pool: descriptor_pool,
            descriptor_layout: descriptor_layout[0]
        }
    }
}

fn render_loop<F: Fn()>(event_loop: &mut EventLoop<()>, f: F) {
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


fn main() {
    let window_height = 800;
    let window_width  = 600;
    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Reforge")
        .with_inner_size(winit::dpi::LogicalSize::new(
            f64::from(window_height),
            f64::from(window_width),
        ))
        .build(&event_loop)
        .unwrap();

    unsafe {
    let res = VkRes::new(&event_loop, &window);

    render_loop(&mut event_loop, || {
        static mut FRAME_INDEX: u8 = 0;

        let frame = &res.frames[FRAME_INDEX as usize];

        /*
        let (present_index, _) = res
            .swapchain.loader
            .acquire_next_image(
                res.swapchain.vk,
                std::u64::MAX,
                frame.present_complete_semaphore,
                vk::Fence::null(),
            )
            .unwrap();


        let wait_semaphors = [frame.render_complete_semaphore];
        let swapchains = [res.swapchain.vk];
        let image_indices = [present_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&wait_semaphors) // &base.rendering_complete_semaphore)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        res.swapchain.loader
            .queue_present(res.queue, &present_info)
            .unwrap();
        */

        FRAME_INDEX = (FRAME_INDEX+1)%NUM_FRAMES;
    });

    }



}
