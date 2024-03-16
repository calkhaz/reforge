extern crate ash;
extern crate gpu_allocator;

use gpu_allocator as gpu_alloc;
use gpu_allocator::vulkan as gpu_alloc_vk;

use ash::vk;
use ash::vk::Handle;
use spirv_reflect::types::ReflectDescriptorBinding;
use spirv_reflect::types::ReflectDescriptorType;
use spirv_reflect::types::ReflectShaderStageFlags;
use crate::Config;
use crate::ConfigDescriptor;
use crate::vulkan::core::VkCore;
use crate::vulkan::shader::ShaderBindings;
use crate::vulkan::shader::Shader;
use crate::vulkan::pipeline::PipelineInfo;
use crate::warnln;

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;


pub struct Sampler {
    device: Rc<ash::Device>,
    pub vk: vk::Sampler
}

pub struct Image {
    device: Rc<ash::Device>,
    allocator: Rc<RefCell<gpu_alloc_vk::Allocator>>,
    pub allocation: gpu_alloc_vk::Allocation,
    pub vk: vk::Image,
    pub view: Option<vk::ImageView>,
    pub format: vk::Format
}

pub struct Buffer {
    device: Rc<ash::Device>,
    allocator: Rc<RefCell<gpu_alloc_vk::Allocator>>,
    pub allocation: gpu_alloc_vk::Allocation,
    pub vk: vk::Buffer,
    pub mapped_data: *mut u8
}

pub struct GpuTimer {
    device: Rc<ash::Device>,
    pub query_pool: vk::QueryPool,
    query_indices: HashMap<String, u32>,
    current_query_index: u32,
    pub query_pool_size: u32,
    pub last_times: HashMap<String, f32>
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
            device,
            query_pool,
            query_indices: HashMap::new(),
            current_query_index: 0,
            query_pool_size: count,
            last_times: HashMap::new()
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

    /*
    pub fn set_elapsed_ms(&mut self) {
        let mut times: HashMap<String, f32> = HashMap::new();

        if self.current_query_index == 0 {
            return;
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
            times.insert(name.clone(), ms);
        }

        self.last_times =  times

    }
    */
}

/* Take the parsed configuration and read the shader of each corresponding pipeline
 * We then match up the bindings parsed from the spirv of the shader and the
 * configuration to create the PipelineInfo(s) */
pub fn synthesize_config(device: Rc<ash::Device>, config: &Config) -> Option<HashMap<String, PipelineInfo>> {
    let mut infos: HashMap<String, PipelineInfo> = HashMap::new();

    for (pipeline_name, pipeline) in &config.graph_pipelines {
        let shader = Shader::from_path(&device, &pipeline.file_path)?;

        let mut info = PipelineInfo {
            name: pipeline_name.clone(),
            shader: Rc::new(RefCell::new(shader)),
            input_images: Vec::new(), output_images: Vec::new(),
            input_ssbos : Vec::new(), output_ssbos : Vec::new()
        };

        // Match up the parsed configuration with what the parsed spirv bindings of the shader
        let add_resource_and_descriptor = |file_path: &str, config_bindings: &Vec<ConfigDescriptor> |
                                           -> Option<(Vec<(String, ReflectDescriptorBinding)>, Vec<(String, ReflectDescriptorBinding)>)> {
            let mut image_bindings : Vec<(String, ReflectDescriptorBinding)> = Vec::new();
            let mut buffer_bindings: Vec<(String, ReflectDescriptorBinding)> = Vec::new();

            for config_binding in config_bindings {
                match info.shader.borrow().bindings.images.get(&config_binding.descriptor_name) {
                    // Try to find a matching image descriptor
                    Some(binding) => {
                        image_bindings.push((config_binding.resource_name.clone(), binding.clone()));
                    },
                    None => match info.shader.borrow().bindings.ssbos.get(&config_binding.descriptor_name) {

                        // Try to find a matching buffer descriptor if no image descriptor was found
                        Some(binding) => {
                            buffer_bindings.push((config_binding.resource_name.clone(), binding.clone()));
                        }
                        None => {
                            // We use the same "-> output" syntax for fragment shaders, but they
                            // output to a framebuffer rather than always needing an output_image.
                            // This isn't the prettiest solution, but it does work for now
                            if info.shader.borrow().stage == vk::ShaderStageFlags::FRAGMENT && config_binding.descriptor_name == "output_image" {
                                continue;
                            }
                            warnln!("Shader {file_path} has no binding named: {}", config_binding.descriptor_name);
                            return None
                        }
                    }
                };
            }

            Some((image_bindings, buffer_bindings))
        };

        (info.input_images , info.input_ssbos)  = add_resource_and_descriptor(&pipeline.file_path, &pipeline.inputs)?;
        (info.output_images, info.output_ssbos) = add_resource_and_descriptor(&pipeline.file_path, &pipeline.outputs)?;

        infos.insert(pipeline_name.clone(), info);
    }

    Some(infos)
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

    let mapped_data : *mut u8 = if mem_type == gpu_alloc::MemoryLocation::GpuToCpu {
        allocation.mapped_ptr().unwrap().as_ptr() as *mut u8
    }
    else {
        std::ptr::null_mut()
    };

    core.set_debug_name(&name, buffer.as_raw(), vk::ObjectType::BUFFER);

    Buffer{device: Rc::clone(&core.device), allocator: Rc::clone(&allocator), vk: buffer, allocation: allocation, mapped_data: mapped_data}
}

// TODO: Don't be lazy about usage here
pub unsafe fn create_image(core: &VkCore, name: String, format: vk::Format, width: u32, height: u32) -> Image {

    let mut usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED;

    // SRGB will not be used in the compute shaders because
    // vulkan complained about it and didn't allow it
    if format != vk::Format::R8G8B8A8_SRGB {
        usage |= vk::ImageUsageFlags::STORAGE;
        usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
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
        // TODO: Optimize for what is actually needed
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

    core.set_debug_name(&name, vk_image.as_raw(), vk::ObjectType::IMAGE);

    Image {
        device: Rc::clone(&core.device),
        format: format,
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

pub fn reflect_stage_to_vk(desc_type: ReflectShaderStageFlags) -> Option<vk::ShaderStageFlags> {
    match desc_type {
        ReflectShaderStageFlags::VERTEX   => Some(vk::ShaderStageFlags::VERTEX),
        ReflectShaderStageFlags::FRAGMENT => Some(vk::ShaderStageFlags::FRAGMENT),
        ReflectShaderStageFlags::COMPUTE  => Some(vk::ShaderStageFlags::COMPUTE),
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

pub unsafe fn create_sampler(device: Rc<ash::Device>) -> Sampler {
    let sampler_info = vk::SamplerCreateInfo::builder()
        .min_filter(vk::Filter::LINEAR)
        .mag_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

    let vk = device.create_sampler(&sampler_info, None).unwrap();

    Sampler { device, vk }
}

pub unsafe fn build_vertex_shader(device: &Rc<ash::Device>) -> Shader {
    // full-screen triangle
    let vertex_shader_code = r#"
        #version 450

        vec2 positions[3] = vec2[](
            vec2(-1.0, -3.0), // top left
            vec2(-1.0,  1.0), // bottom left
            vec2( 3.0 , 1.0)  // bottom right
        );

        void main() {
            gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
        }
    "#;

    Shader::from_contents(device, "full-screen-triangle".to_string(), vk::ShaderStageFlags::VERTEX, vertex_shader_code.to_string()).unwrap()
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

impl Drop for Sampler {
    fn drop(&mut self) { unsafe { self.device.destroy_sampler(self.vk, None); } }
}

impl Drop for GpuTimer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_query_pool(self.query_pool, None);
        }
    }
}
