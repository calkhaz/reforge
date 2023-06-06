extern crate ash;
extern crate gpu_allocator;

use gpu_allocator as gpu_alloc;
use gpu_allocator::vulkan as gpu_alloc_vk;

use ash::vk;
use crate::vulkan::core::VkCore;

use std::rc::Rc;
use std::cell::RefCell;

pub struct Image {
    device: Rc<ash::Device>,
    allocator: Rc<RefCell<gpu_alloc_vk::Allocator>>,
    pub allocation: gpu_alloc_vk::Allocation,
    pub vk: vk::Image,
    pub view: Option<vk::ImageView>,
}

pub struct Buffer {
    device: Rc<ash::Device>,
    allocator: Rc<RefCell<gpu_alloc_vk::Allocator>>,
    pub allocation: gpu_alloc_vk::Allocation,
    pub vk: vk::Buffer
}

pub unsafe fn create_buffer(core: &VkCore,
                            name : String,
                            size: vk::DeviceSize,
                            usage: vk::BufferUsageFlags,
                            mem_type: gpu_alloc::MemoryLocation) -> Buffer {
    let info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = core.device.create_buffer(&info, None).unwrap();

    let allocator = &core.allocator.as_ref().unwrap();
    let allocation = allocator.borrow_mut()
        .allocate(&gpu_alloc_vk::AllocationCreateDesc {
            requirements: core.device.get_buffer_memory_requirements(buffer),
            location: mem_type,
            linear: true,
            allocation_scheme: gpu_alloc_vk::AllocationScheme::GpuAllocatorManaged,
            name: &name
        })
        .unwrap();

    core.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()).unwrap();

    Buffer{device: Rc::clone(&core.device), allocator: Rc::clone(&allocator), vk: buffer, allocation: allocation}
}

pub unsafe fn create_image(core: &VkCore, name: String, format: vk::Format, width: u32, height: u32) -> Image {

    let mut usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;

    // SRGB will not be used in the compute shaders because
    // vulkan complained about it and didn't allow it
    if format != vk::Format::R8G8B8A8_SRGB {
        usage |= vk::ImageUsageFlags::STORAGE;
    }

    let input_image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .format(format)
        .mip_levels(1)
        .extent(vk::Extent3D{width: width, height: height, depth: 1})
        // TOOD: Optimize for what is actually needed
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let vk_image = core.device.create_image(&input_image_info, None).unwrap();

    let allocator = &core.allocator.as_ref().unwrap();
    let image_allocation = allocator.borrow_mut()
        .allocate(&gpu_alloc_vk::AllocationCreateDesc {
            requirements: core.device.get_image_memory_requirements(vk_image),
            location: gpu_alloc::MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: gpu_alloc_vk::AllocationScheme::GpuAllocatorManaged,
            name: &name
        })
        .unwrap();


    core.device.bind_image_memory(vk_image, image_allocation.memory(), image_allocation.offset()).unwrap();

    // Because SRGB is only going to be used for blitting, an image view
    // is not required and only causes validation warnings
    let image_view = if format != vk::Format::R8G8B8A8_SRGB {
        let image_view_info = vk::ImageViewCreateInfo::builder()
            .image(vk_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(*vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1));

        Some(core.device.create_image_view(&image_view_info, None).unwrap())
    }
    else {
        None
    };

    Image {
        device: Rc::clone(&core.device),
        allocator: Rc::clone(&allocator),
        vk: vk_image,
        view: image_view,
        allocation: image_allocation
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.vk, None);
        }
        let allocation = std::mem::take(&mut self.allocation);
        self.allocator.borrow_mut().free(allocation).unwrap();
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            if let Some(view) = self.view {
                self.device.destroy_image_view(view, None);
            }
            self.device.destroy_image(self.vk, None);
        }
        let allocation = std::mem::take(&mut self.allocation);
        self.allocator.borrow_mut().free(allocation).unwrap();
    }
}
