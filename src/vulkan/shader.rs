extern crate spirv_reflect;
use ash::vk;
use shaderc::CompilationArtifact;
use spirv_reflect::types::{ReflectDescriptorType, ReflectDescriptorBinding};

use std::collections::HashMap;

use crate::warnln;

#[derive(Debug)]
pub struct ShaderBindings {
    pub images: HashMap<String, ReflectDescriptorBinding>,
    pub buffers: Vec<ReflectDescriptorBinding>,
    pub ssbos: HashMap<String, ReflectDescriptorBinding>
}

pub struct Shader {
    pub module: vk::ShaderModule,
    pub bindings: ShaderBindings,
    pub path: String
}

impl Shader {
    pub fn new(device: &ash::Device, path: &str) -> Option<Shader> {
        // Compile glsl to spirv
        let spirv_artifact = Self::create_spirv(&path)?;
        let spirv_binary : &[u32] = spirv_artifact.as_binary();

        Some(Shader {
            module  : Self::create_module(device, spirv_binary)?,
            bindings: Self::reflect_descriptors(spirv_binary)?,
            path: path.to_string()
        })
    }

    fn create_module(device: &ash::Device, spirv_binary: &[u32]) -> Option<vk::ShaderModule> {

        let shader_info = vk::ShaderModuleCreateInfo::builder().code(spirv_binary);

        unsafe {
        match device.create_shader_module(&shader_info, None) {
            Ok(module) => Some(module),
            Err(e) => { warnln!("{:?}", e); None }
        }
        }
    }

    fn create_spirv(path: &str) -> Option<CompilationArtifact> {
        let glsl_source = match std::fs::read_to_string(path) {
            Ok(contents) => Some(contents),
            Err(err) => { warnln!("Failed to read the file '{path}': {}", err); None }
        }?;

        if glsl_source.is_empty() {
            warnln!("File was empty: {path}");
            return None
        }

        let compiler = shaderc::Compiler::new().unwrap();
        let options = shaderc::CompileOptions::new().unwrap();

        match compiler.compile_into_spirv(&glsl_source.to_owned(),
                                          shaderc::ShaderKind::Compute,
                                          path,
                                          "main",
                                          Some(&options)) {
            Ok(binary) => {
                assert_eq!(Some(&0x07230203), binary.as_binary().first());

                Some(binary)
            } ,

            // Remove extra newline from error before printing
            // Remove the "compilation error:\n" before a single error, which is not very useful
            // On multiple errors, it may say "2 compilation errors:", which can be useful
            Err(e) => { warnln!("{}", e.to_string().trim_start_matches("compilation error:\n").trim_end_matches('\n')); None }
        }
    }

    fn reflect_desc_to_vk(desc_type: ReflectDescriptorType) -> Option<vk::DescriptorType> {
        match desc_type {
            ReflectDescriptorType::StorageImage         => Some(vk::DescriptorType::STORAGE_IMAGE),
            ReflectDescriptorType::CombinedImageSampler => Some(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            ReflectDescriptorType::Sampler              => Some(vk::DescriptorType::SAMPLER),
            ReflectDescriptorType::UniformBuffer        => Some(vk::DescriptorType::UNIFORM_BUFFER),
            ReflectDescriptorType::StorageBuffer        => Some(vk::DescriptorType::STORAGE_BUFFER),
            _ => None
        }
    }

    fn reflect_descriptors(binary: &[u32]) ->  Option<ShaderBindings> {
        let module = match spirv_reflect::ShaderModule::load_u32_data(binary) {
            Ok(module) => Some(module),
            Err(e) => { warnln!("{:?}", e); None }
        }?;

        let sets = match module.enumerate_descriptor_sets(None) {
            Ok(sets) => { Some(sets) } ,
            Err(err) => {warnln!("{:?}", err); None }
        }?;

        // The bindings for the pipeline
        let mut images : HashMap<String, ReflectDescriptorBinding> = HashMap::new();
        let mut buffers : Vec<ReflectDescriptorBinding> = Vec::new();
        let mut ssbos : HashMap<String, ReflectDescriptorBinding> = HashMap::new();

        if sets.len() > 1 {
            warnln!("Warning: Cannot currently handle more than one descriptor set per shader");
        }

        for set in &sets {
            for binding in &set.bindings {
                if set.set > 0 {
                    panic!("Currently only support a single descriptor at idx 0 set per shader");
                }

                let desc_is_buffer = |desc_type: ReflectDescriptorType| -> bool {
                    desc_type == ReflectDescriptorType::UniformBuffer || desc_type == ReflectDescriptorType::StorageBuffer
                };

                let desc_type = Self::reflect_desc_to_vk(binding.descriptor_type);

                if desc_type.is_none() {
                    warnln!("Warning: Unable to handle descriptor type '{:?}'", desc_type); 
                    break;
                }

                if desc_is_buffer(binding.descriptor_type) {
                    if binding.descriptor_type == ReflectDescriptorType::StorageBuffer {
                        // Grab ssbo by the block name
                        ssbos.insert(binding.type_description.as_ref().unwrap().type_name.clone(), binding.clone());
                    }
                    buffers.push(binding.clone());
                }
                // Images are always top level and have names, just insert them
                else {
                    images.insert(binding.name.clone(), binding.clone());
                }
            }
        }

        Some(ShaderBindings {
            images: images,
            buffers: buffers,
            ssbos: ssbos
        })
    }
}
