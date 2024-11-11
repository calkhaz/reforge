extern crate ash;

use anyhow::{Result, Context};
use ash::vk;
use ash::khr;

use std::rc::Rc;

use crate::vulkan::core::VkCore;

pub struct SwapChain {
    pub vk: vk::SwapchainKHR,
    pub loader: khr::swapchain::Device,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    device: Rc<ash::Device>,
    pub width: u32,
    pub height: u32
}

impl SwapChain {
    pub unsafe fn new(core: &VkCore, width: u32, height: u32) -> Result<SwapChain> {
        let swapchain_loader = khr::swapchain::Device::new(&core.instance, &core.device);
        let (swapchain, surface_format) = SwapChain::build(core, width, height, &swapchain_loader, None)?;

        let images = swapchain_loader.get_swapchain_images(swapchain)?;
        let views = SwapChain::create_present_image_views(core, &images, surface_format)?;

        Ok(SwapChain {
            device: Rc::clone(&core.device),
            vk: swapchain,
            loader: swapchain_loader,
            images,
            views,
            width,
            height
        })
    }

    unsafe fn create_present_image_views(core: &VkCore, images: &Vec<vk::Image>, surface_format: vk::SurfaceFormatKHR) -> Result<Vec<vk::ImageView>> {
        let res: Result<Vec<_>, _> = images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo::default()
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
                core.device.create_image_view(&create_view_info, None)
            }).collect();

        Ok(res?)
    }

    unsafe fn build(core: &VkCore, width: u32, height: u32, swapchain_loader: &khr::swapchain::Device, old_swapchain: Option<vk::SwapchainKHR>) -> Result<(vk::SwapchainKHR, vk::SurfaceFormatKHR)> {
        let surface = core.surface.expect("Cannot create swapchain without a valid surface");
        let surface_loader = core.surface_loader.as_ref().expect("Cannot create swapchain without a valid surface loader");

        let surface_capabilities = surface_loader
            .get_physical_device_surface_capabilities(core.pdevice, surface)?;

        let surface_format = surface_loader
            .get_physical_device_surface_formats(core.pdevice, surface)
            .expect("Found no valid formats for the swapchain")
            .iter().cloned()
            .find(|&format| format.format == vk::Format::B8G8R8A8_SRGB &&
                            format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .context("Could not find desired swapchain B8G8R8A8_SRGB format")?;

        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }
        let surface_resolution = match surface_capabilities.current_extent.width {
            std::u32::MAX => vk::Extent2D { width, height },
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
            .get_physical_device_surface_present_modes(core.pdevice, surface)?;

        let present_mode = present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(desired_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_resolution)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1)
            .old_swapchain(old_swapchain.unwrap_or(vk::SwapchainKHR::default()));

        Ok((swapchain_loader.create_swapchain(&swapchain_create_info, None)?, surface_format))
    }

    pub unsafe fn rebuild(&mut self, core: &VkCore, width: u32, height: u32) -> Result<()> {
        for &view in &self.views {
            self.device.destroy_image_view(view, None);
        }
        self.views.clear();
        self.images.clear();

        let (swapchain, surface_format) = Self::build(core, width, height, &self.loader, Some(self.vk))?;

        self.vk = swapchain;
        self.images = self.loader.get_swapchain_images(swapchain)?;
        self.views = SwapChain::create_present_image_views(core, &self.images, surface_format)?;
        self.width = width;
        self.height = height;
        Ok(())
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
