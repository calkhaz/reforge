extern crate ash;
extern crate gpu_allocator;

use gpu_allocator as gpu_alloc;
use gpu_allocator::vulkan as gpu_alloc_vk;

use ash::vk;

pub struct Image {
    pub vk: vk::Image,
    pub view: vk::ImageView,
    pub allocation: gpu_alloc_vk::Allocation
}

pub struct Buffer {
    pub vk: vk::Buffer,
    pub allocation: gpu_alloc_vk::Allocation
}

pub unsafe fn create_buffer(device: &ash::Device,
                            size: vk::DeviceSize,
                            usage: vk::BufferUsageFlags,
                            mem_type: gpu_alloc::MemoryLocation,
                            allocator: &mut gpu_alloc_vk::Allocator) -> Buffer {
    let info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&info, None).unwrap();

    let allocation = allocator
        .allocate(&gpu_alloc_vk::AllocationCreateDesc {
            requirements: device.get_buffer_memory_requirements(buffer),
            location: mem_type,
            linear: true,
            allocation_scheme: gpu_alloc_vk::AllocationScheme::GpuAllocatorManaged,
            name: "input-image-staging-buffer",
        })
        .unwrap();

    device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()).unwrap();

    Buffer{vk: buffer, allocation: allocation}
}

pub unsafe fn create_image(device: &ash::Device, name: String, width: u32, height: u32, allocator: &mut gpu_alloc_vk::Allocator) -> Image {
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

    let vk_image = device.create_image(&input_image_info, None).unwrap();

    let image_allocation = allocator
        .allocate(&gpu_alloc_vk::AllocationCreateDesc {
            requirements: device.get_image_memory_requirements(vk_image),
            location: gpu_alloc::MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: gpu_alloc_vk::AllocationScheme::GpuAllocatorManaged,
            name: &name
        })
        .unwrap();


    device.bind_image_memory(vk_image, image_allocation.memory(), image_allocation.offset()).unwrap();

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

    let image_view = device.create_image_view(&image_view_info, None).unwrap();

    Image {
        vk: vk_image,
        view: image_view,
        allocation: image_allocation
    }
}

