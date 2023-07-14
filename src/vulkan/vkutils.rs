extern crate ash;
extern crate gpu_allocator;

use gpu_allocator as gpu_alloc;
use gpu_allocator::vulkan as gpu_alloc_vk;

use ash::vk;
use spirv_reflect::types::ReflectDescriptorBinding;
use spirv_reflect::types::ReflectDescriptorType;
use crate::vulkan::core::VkCore;
use crate::vulkan::shader::ShaderBindings;
use crate::vulkan::pipeline_graph::PipelineInfo;

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::HashMap;

use crate::config::config::ConfigPipeline;

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

pub struct GpuTimer {
    device: Rc<ash::Device>,
    pub query_pool: vk::QueryPool,
    query_indices: BTreeMap<String, u32>, // BTreeMap because we want names sorted for printing
    current_query_index: u32,
    pub query_pool_size: u32
}

impl GpuTimer{
    pub fn new(device: Rc<ash::Device>, mut count: u32) -> GpuTimer {
        // 2* because need begin + end timers
        count *= 2;

        let query_info = vk::QueryPoolCreateInfo::builder()
            .query_count(count)
            .query_type(vk::QueryType::TIMESTAMP);

        unsafe {
        let query_pool = device.create_query_pool(&query_info, None).unwrap();

        GpuTimer {
            device: device,
            query_pool: query_pool,
            query_indices: BTreeMap::new(),
            current_query_index: 0,
            query_pool_size: count
        }
        }
    }

    pub fn start(&mut self, name: &str) -> u32 {

        // Reserve a beg/end set of query indices if we haven't seen this timer name before
        if !self.query_indices.contains_key(name) {
            // We check +2 because we need 2 slots per timer (start & stop)
            if self.current_query_index + 2 > self.query_pool_size {
                panic!("Ran out of query-pool indices when trying to add timer {}", name);
            }

            self.query_indices.insert(name.to_string(), self.current_query_index);

            // Begin + end
            self.current_query_index += 2;
        }

        *self.query_indices.get(name).unwrap()
    }

    pub fn stop(&mut self, name: &str) -> u32{
        if !self.query_indices.contains_key(name) {
            panic!("No gpu-timer was ever stated with the name: {}", name);
        }

        // Get the ending timer for the name
        *self.query_indices.get(name).unwrap() + 1
    }

    pub fn get_elapsed_ms(&mut self) -> String {
        let mut times: String = String::new();

        if self.current_query_index == 0 {
            return times;
        }

        let mut timestamps: Vec<u64> = vec![0; self.current_query_index as usize];

        unsafe {
        self.device.get_query_pool_results(
            self.query_pool,
            0,                              // first-query-idx
            self.current_query_index,       // last-query-idx
            timestamps.as_mut_slice(),      // data to fill with timestamps
            vk::QueryResultFlags::TYPE_64).unwrap();
        }

        for (name, idx) in &self.query_indices {
            let nanosec_diff = timestamps[*idx as usize+1] - timestamps[*idx as usize];
            let ms = (nanosec_diff as f32)/1e6;

            times.push_str(&format!("{}: {:.3}ms, ", name.to_string(), ms));
        }

        // Remove last ", "
        times.truncate(times.len()-2);

        times

    }
}

pub fn synthesize_config(device: Rc<ash::Device>, config: &HashMap<String, ConfigPipeline>) -> HashMap<String, PipelineInfo> {

    use crate::vulkan::pipeline_graph::{SWAPCHAIN_OUTPUT, FILE_INPUT};
    use crate::vulkan::shader::Shader;

    let mut infos: HashMap<String, PipelineInfo> = HashMap::new();

    for (name, data) in config {
        let shader_path = &data.shader_path;


        let shader = Shader::new(&device, shader_path).expect(&format!("Failed to create shader: {shader_path}"));

        let mut info = PipelineInfo {
            shader: shader,
            input_images: Vec::new(), output_images: Vec::new(),
            input_ssbos : Vec::new(), output_ssbos : Vec::new()
        };

        for image in &data.input_images {
            let desc = info.shader.bindings.images.get(&image.descriptor_name).expect(&format!("Shader {shader_path} has no binding named: {}", image.descriptor_name));
            let resource_name = if image.resource_name == "input_image" { FILE_INPUT.to_string() } else { image.resource_name.clone() };
            info.input_images.push((resource_name, desc.clone()));
        }

        for image in &data.output_images {
            let desc = info.shader.bindings.images.get(&image.descriptor_name).expect(&format!("Shader {shader_path} has no binding named: {}", image.descriptor_name));
            let resource_name = if image.resource_name == "output_image" { SWAPCHAIN_OUTPUT.to_string() } else { image.resource_name.clone() };
            info.output_images.push((resource_name, desc.clone()));
        }

        for buffer in &data.input_buffers {
            let desc = info.shader.bindings.ssbos.get(&buffer.descriptor_name).expect(&format!("Shader {shader_path} has no binding named: {}", buffer.descriptor_name));
            info.input_ssbos.push((buffer.resource_name.clone(), desc.clone()));
        }

        for buffer in &data.output_buffers {
            let desc = info.shader.bindings.ssbos.get(&buffer.descriptor_name).expect(&format!("Shader {shader_path} has no binding named: {}", buffer.descriptor_name));
            info.output_ssbos.push((buffer.resource_name.clone(), desc.clone()));
        }

        infos.insert(name.clone(), info);
    }

    infos
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

pub fn reflect_desc_to_vk(desc_type: ReflectDescriptorType) -> Option<vk::DescriptorType> {
    match desc_type {
        ReflectDescriptorType::StorageImage         => Some(vk::DescriptorType::STORAGE_IMAGE),
        ReflectDescriptorType::CombinedImageSampler => Some(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        ReflectDescriptorType::Sampler              => Some(vk::DescriptorType::SAMPLER),
        ReflectDescriptorType::UniformBuffer        => Some(vk::DescriptorType::UNIFORM_BUFFER),
        ReflectDescriptorType::StorageBuffer        => Some(vk::DescriptorType::STORAGE_BUFFER),
        _ => None
    }
}

//pub fn create_descriptor_layout_bindings(bindings: &HashMap<String, ReflectDescriptorBinding>,
pub fn create_descriptor_layout_bindings(bindings: &ShaderBindings,
                                         num_frames: usize,
                                         pool_sizes: &mut HashMap<vk::DescriptorType, u32>) -> Vec<vk::DescriptorSetLayoutBinding> {

    let mut vk_bindings: Vec<vk::DescriptorSetLayoutBinding> = Vec::with_capacity(bindings.images.len() + bindings.buffers.len());

    let mut image_binding = vk::DescriptorSetLayoutBinding {
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        ..Default::default()
    };

    let mut add_binding = |binding: &ReflectDescriptorBinding| {
        let desc_type = reflect_desc_to_vk(binding.descriptor_type).expect(&format!("Can\'t handle descriptor type: {:?}", binding.descriptor_type));
        // Add to pool size
        *pool_sizes.entry(desc_type).or_insert(0) += num_frames as u32;

        // Add vulkan descriptor binding
        image_binding.binding = binding.binding;
        image_binding.descriptor_type = desc_type;
        vk_bindings.push(image_binding);
    };

    for (_, binding) in &bindings.images {
        add_binding(binding);
    }

    for binding in &bindings.buffers {
        add_binding(binding);
    }

    vk_bindings
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

impl Drop for GpuTimer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_query_pool(self.query_pool, None);
        }
    }
}
