extern crate ash;
extern crate gpu_allocator;


use ash::{vk::{self }};
use std::{ffi::CStr};
use ash::extensions::khr;
use std::borrow::Cow;
use std::default::Default;
use std::os::raw::c_char;
use std::rc::Rc;
use std::cell::RefCell;
use winit::window::Window;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use gpu_allocator::vulkan as gpu_alloc_vk;

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


pub struct VkCore {
    // Allocator is always created, but we need to be able to 
    // move this out of a mutable reference during drop and gpu_alloc_vk::Allocator
    // has no default, so Option<> was the workaround
    pub allocator: Option<Rc<RefCell<gpu_alloc_vk::Allocator>>>,
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: Rc<ash::Device>,
    pub pdevice: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub debug_utils_loader: ash::extensions::ext::DebugUtils,
    pub debug_callback: ash::vk::DebugUtilsMessengerEXT,
    pub surface: Option<vk::SurfaceKHR>,
    pub surface_loader: Option<khr::Surface>
}


impl VkCore {
    pub unsafe fn new(window: &Option<Window>) -> Self {
        let mut extension_names: Vec<*const i8> = Vec::new();
        if window.is_some() {
            extension_names.extend(ash_window::enumerate_required_extensions(window.as_ref().unwrap().raw_display_handle()).unwrap().to_vec());
        }
        extension_names.push(ash::extensions::ext::DebugUtils::name().as_ptr());

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extension_names.push(ash::vk::KhrPortabilityEnumerationFn::name().as_ptr());
            //// Enabling this extension is a requirement when using `VK_KHR_portability_subset`
            extension_names.push(ash::vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr());
        }


        let entry = ash::Entry::load().unwrap();
        let instance = Self::create_instance(&entry, &extension_names);
        let (debug_callback, debug_utils) = Self::create_debug_utils(&instance, &entry);

        let (surface, surface_loader) = match window {
            Some(window) => {
                let (surface, surface_loader) = Self::create_surface(&entry, &instance, &window);
                (Some(surface), Some(surface_loader))
            }
            None => (None, None)
        };

        let (pdevice, queue_family_index) = Self::create_physical_device(&instance, surface, &surface_loader);

        let mut device_extension_names_raw : Vec<*const i8> = vec![
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ash::vk::KhrPortabilitySubsetFn::name().as_ptr(),
        ];

        if window.is_some() {
            device_extension_names_raw.push(khr::Swapchain::name().as_ptr());
        }

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

        let queue = device.get_device_queue(queue_family_index, 0);

        let allocator = Rc::new(RefCell::new(gpu_alloc_vk::Allocator::new(&gpu_alloc_vk::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: pdevice,
            debug_settings: Default::default(),
            buffer_device_address: false,
        }).unwrap()));

        VkCore {
            entry: entry,
            instance: instance,
            device: Rc::new(device),
            pdevice: pdevice,
            queue: queue,
            queue_family_index: queue_family_index,
            debug_utils_loader: debug_utils,
            debug_callback: debug_callback,
            surface: surface,
            surface_loader: surface_loader,
            allocator: Some(allocator)
        }
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

    unsafe fn create_physical_device(instance: &ash::Instance, surface: Option<vk::SurfaceKHR>, surface_loader: &Option<khr::Surface>) -> (vk::PhysicalDevice, u32) {
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
                        let mut supports_graphic_and_surface = info.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE);

                        if surface.is_some() {
                            supports_graphic_and_surface &= surface_loader.as_ref().unwrap().get_physical_device_surface_support(
                                *pdevice,
                                index as u32,
                                surface.unwrap(),
                            ).unwrap();
                        }

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

}

impl Drop for VkCore {
    fn drop(&mut self) {
        unsafe {
            // unwrap option and move the data from the struct
            let allocator_opt = std::mem::take(&mut self.allocator).unwrap();

            // Pull the allocator out of the rc and refcell
            let allocator = Rc::try_unwrap(allocator_opt).unwrap().into_inner();

            // Drop allocator before device destruction
            drop(allocator);

            self.device.device_wait_idle().unwrap();
            self.device.destroy_device(None);
            if self.surface.is_some() {
                self.surface_loader.as_ref().unwrap().destroy_surface(self.surface.unwrap(), None);
            }
            self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}
