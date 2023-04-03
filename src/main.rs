extern crate ash;
use ash::{vk, Entry};
pub use ash::{Device, Instance};

use std::ffi::CStr;

use ash::extensions::{
    ext::DebugUtils,
    khr::{Surface, Swapchain},
};
use std::borrow::Cow;
use std::cell::RefCell;
use std::default::Default;
use std::ops::Drop;
use std::os::raw::c_char;

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

pub struct VkRes {
    pub entry: Entry,

    pub instance: Instance,
    pub device: Device,
//    pub surface_loader: Surface,
//    pub swapchain_loader: Swapchain,
//    pub debug_utils_loader: DebugUtils,
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
    unsafe fn create_debug_utils(instance : &Instance, entry: &Entry) -> vk::DebugUtilsMessengerEXT {
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

        let debug_utils_loader = DebugUtils::new(&entry, &instance);

        let debug_call_back = debug_utils_loader
            .create_debug_utils_messenger(&debug_info, None)
            .unwrap();

        debug_call_back
    }

    unsafe fn create_instance(entry: &Entry) -> Instance {
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

        let extension_names = vec![DebugUtils::name().as_ptr()];
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

        let entry = Entry::load().unwrap();

        let instance: Instance = entry
            .create_instance(&create_info, None)
            .expect("Instance creation error");


        instance
    }

    unsafe fn new() -> Self{
        let entry = Entry::load().unwrap();

        let instance = Self::create_instance(&entry);
        let debug_utils = Self::create_debug_utils(&instance, &entry);

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
                        let supports_compute = info.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE);

                        if supports_compute {
                            Some((*pdevice, index))
                        } else {
                            None
                        }
                    })
            })
            .expect("Couldn't find suitable device.");


        let queue_family_index = queue_family_index as u32;

        let device_extension_names_raw : [*const i8; 1] = [
            Swapchain::name().as_ptr(),
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
//            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&features);

        let device: Device = instance
            .create_device(pdevice, &device_create_info, None)
            .unwrap();

//        let present_queue = device.get_device_queue(queue_family_index, 0);
    
        VkRes {
            entry: entry,
            instance: instance,
            device: device
        }
    }

}

fn main() {
    unsafe {
    let res = VkRes::new();
    }
}
