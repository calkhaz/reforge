extern crate spirv_reflect;
use ash::vk;
use shaderc::CompilationArtifact;
use spirv_reflect::types::ReflectDescriptorType;

use std::collections::HashMap;

pub struct DescriptorBinding {
    pub descriptor_type: vk::DescriptorType,
    pub index: u32
}

pub struct Shader {
    pub module: vk::ShaderModule,
    pub bindings: HashMap<String, DescriptorBinding>
}

impl Shader {
    pub fn new(device: &ash::Device, path: &str) -> Option<Shader> {
        // Compile glsl to spirv
        let spirv_artifact = Self::create_spirv(&path)?;
        let spirv_binary : &[u32] = spirv_artifact.as_binary();

        Some(Shader {
            module  : Self::create_module(device, spirv_binary)?,
            bindings: Self::reflect_descriptors(spirv_binary)?
        })
    }

    fn create_module(device: &ash::Device, spirv_binary: &[u32]) -> Option<vk::ShaderModule> {

        let shader_info = vk::ShaderModuleCreateInfo::builder().code(spirv_binary);

        unsafe {
        match device.create_shader_module(&shader_info, None) {
            Ok(module) => Some(module),
            Err(e) => { eprintln!("{:?}", e); None }
        }
        }
    }

    fn create_spirv(path: &str) -> Option<CompilationArtifact> {
        let glsl_source = std::fs::read_to_string(path)
            .expect("Should have been able to read the file");

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
            Err(e) => { eprintln!("{:?}", e); None }
        }
    }

    fn reflect_descriptors(binary: &[u32]) ->  Option< HashMap<String, DescriptorBinding> > {
        let module = match spirv_reflect::ShaderModule::load_u32_data(binary) {
            Ok(module) => Some(module),
            Err(e) => { eprintln!("{:?}", e); None }
        }?;

        let sets = match module.enumerate_descriptor_sets(None) {
            Ok(sets) => { Some(sets) } ,
            Err(err) => {eprint!("{:?}", err); None }
        }?;

        // The bindings for the pipeline
        let mut bindings : HashMap<String, DescriptorBinding> = HashMap::new();

        if sets.len() > 1 {
            eprintln!("Warning: Cannot currently handle more than one descriptor set per shader");
        }

        for set in &sets {
            for binding in &set.bindings {
                if set.set > 0 {
                    panic!("Currently only support a single descriptor at idx 0 set per shader");
                }

                let mut desc_type = vk::DescriptorType::STORAGE_IMAGE;

                match binding.descriptor_type {
                    ReflectDescriptorType::StorageImage         =>  desc_type = vk::DescriptorType::STORAGE_IMAGE,
                    ReflectDescriptorType::CombinedImageSampler => desc_type = vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ReflectDescriptorType::Sampler              => desc_type = vk::DescriptorType::SAMPLER,
                    ReflectDescriptorType::UniformBuffer        => desc_type = vk::DescriptorType::UNIFORM_BUFFER,
                    ReflectDescriptorType::StorageBuffer        => desc_type = vk::DescriptorType::STORAGE_BUFFER,
                    _ => { eprintln!("Warning: Unrecognized binding '{}'", binding.name) }
                }

                bindings.insert(binding.name.clone(), DescriptorBinding {
                    descriptor_type: desc_type,
                    index: binding.binding
                });
            }
        }

        Some(bindings)
    }
}
