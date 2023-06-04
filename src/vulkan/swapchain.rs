extern crate ash;

use ash::vk;
use ash::extensions::khr;

use std::rc::Rc;

use crate::vulkan::core::VkCore;

pub struct SwapChain {
    surface_format: vk::SurfaceFormatKHR,
    pub vk: vk::SwapchainKHR,
    pub loader: khr::Swapchain,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    device: Rc<ash::Device>
}

impl SwapChain {
    pub unsafe fn new(core: &VkCore, width: u32, height: u32) -> SwapChain {
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
            device: Rc::clone(&core.device),
            surface_format: surface_format,
            vk: swapchain,
            loader: swapchain_loader,
            images: present_images,
            views: present_image_views
        };
    }
}

impl Drop for SwapChain {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            for &view in &self.views {
                self.device.destroy_image_view(view, None);
            }
            self.loader.destroy_swapchain(self.vk, None);
        }
    }
}
