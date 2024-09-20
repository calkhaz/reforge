use anyhow::{anyhow, Result, Context};
use ash::vk;
use shaderc::CompilationArtifact;
use crate::{utils, err};

use std::rc::Rc;
use std::collections::HashMap;

use crate::warnln;
use bitflags::bitflags;

#[derive(Clone, Copy, Debug)]
pub enum DescType {
    CombinedImageSampler,
    StorageImage,
    UniformBuffer,
    StorageBuffer,
}

impl DescType {
    pub fn to_vk(&self) -> vk::DescriptorType {
        match self {
            DescType::StorageImage         => vk::DescriptorType::STORAGE_IMAGE,
            DescType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            DescType::UniformBuffer        => vk::DescriptorType::UNIFORM_BUFFER,
            DescType::StorageBuffer        => vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

bitflags! {
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DescBlockType: u32 {
    const UNDEFINED = 0;
    const VOID = 1;
    const BOOL = 2;
    const INT = 4;
    const FLOAT = 8;
    const VECTOR = 256;
    const MATRIX = 512;
    const EXTERNAL_IMAGE = 65536;
    const EXTERNAL_SAMPLER = 131_072;
    const EXTERNAL_SAMPLED_IMAGE = 262_144;
    const EXTERNAL_BLOCK = 524_288;
    const EXTERNAL_ACCELERATION_STRUCTURE_NV = 1_048_576;
    const EXTERNAL_MASK = 2_031_616;
    const STRUCT = 268_435_456;
    const ARRAY = 536_870_912;
}
}

#[derive(Clone, Debug)]
pub struct ImageBinding {
    pub binding: u32,
    pub image_type: DescType
}

#[derive(Clone, Debug)]
pub struct SsboBinding {
    pub binding: u32,
    pub size: usize
}

#[derive(Debug)]
pub struct UboBinding {
    pub binding: u32,
    pub ubos: HashMap<String, UboVar>,
    pub size: usize
}

#[derive(Debug)]
pub struct UboVar {
    pub size: usize,
    pub offset: usize,
    pub block_type: DescBlockType,
    pub array_stride: usize,
    pub array_len: u32,
    // Currently vecs are unsupported
    //pub dims: Vec<u32>
}

pub struct ShaderBindings {
    pub images: HashMap<String, ImageBinding>,
    pub ubos: HashMap<String, UboBinding>,
    pub ssbos: HashMap<String, SsboBinding>
}

pub struct Shader {
    device: Rc<ash::Device>,
    pub _name: String,
    pub module: vk::ShaderModule,
    pub bindings: ShaderBindings,
    pub path: Option<String>,
    pub stage: vk::ShaderStageFlags
}
impl UboBinding {
    fn get_scalar_type(s: &spirq::ty::ScalarType) -> Result<DescBlockType> {
        use spirq::ty::ScalarType;

        let t = match s {
            ScalarType::Boolean => DescBlockType::BOOL,
            ScalarType::Float { bits } => {
                if *bits != 32 { return err!("Cannot support float of {} bits", bits) }
                DescBlockType::FLOAT
            },
            ScalarType::Integer { bits, .. } => {
                if *bits != 32 { return err!("Cannot support integer of {} bits", bits) }
                DescBlockType::INT
            },
            _ => return err!("Unsupported scalar type {}", s)
        };

        Ok(t)
    }

    fn get_array_type(v: &spirq::ty::ArrayType) -> Result<(usize, u32, DescBlockType)> {
        use spirq::ty::Type::*;

        let arr_stride = match v.stride {
            Some(s) => s,
            None => return err!("Cannot handle unknown arr stride")
        };
        let arr_len = match v.nelement {
            Some(s) => s, None => return err!("Cannot handle unknown arr len")
        };

        let block_type = match v.element_ty.as_ref() {
            Scalar(v) => Self::get_scalar_type(&v)?,
            _ => return err!("Unsupported array type {}", v)
        } | DescBlockType::ARRAY;

        Ok((arr_stride, arr_len, block_type))
    }

    // Recursively convert struct of UBO into flatten mapping in struct.member.member.member... like format
    fn flatten_ubo(name: String, base_offset: usize, block: &spirq::ty::Type, ubos: &mut HashMap<String, UboVar>) -> Result<()> {
        use spirq::ty::Type::*;

        let size = block.nbyte().context("Ubo block has no size")?;

        match block {
            Scalar(v) => {
                let block_var = UboVar {
                    size,
                    offset: base_offset,
                    block_type: Self::get_scalar_type(v)?,
                    array_stride: 0,
                    array_len: 0,
                };
                ubos.insert(name, block_var);
            },
            Array(v) => {
                let (arr_stride, arr_len, block_type) = Self::get_array_type(v)?;

                let block_var = UboVar {
                    size,
                    offset: base_offset,
                    block_type,
                    array_stride: arr_stride,
                    array_len: arr_len,
                };
                ubos.insert(name, block_var);
            },
            Struct(v) => {
                for mem in &v.members {
                    let mem_offset = match mem.offset {
                        Some(o) => o, None => return err!("Member has no known offset")
                    };

                    let abs_offset = mem_offset + base_offset;

                    let mem_name = match &mem.name {
                        Some(o) => o, None => return err!("Member has no name")
                    };

                    let full_name = if name.is_empty() { mem_name.clone() }
                    else { format!("{name}.{mem_name}") };

                    Self::flatten_ubo(full_name, abs_offset, &mem.ty, ubos)?;
                }
            },
            _ => return err!("Unsupported block type {}", block)
        };

        Ok(())
    }

    pub fn new(name_prefix: String,
               desc_bind: &spirq::var::DescriptorBinding,
               block: &spirq::ty::Type) -> Result<UboBinding> {

        let mut ubos: HashMap<String, UboVar> = HashMap::new();
        Self::flatten_ubo(name_prefix, 0, block, &mut ubos)?;

        let size = block.min_nbyte().context("Ubo block has no size")?;

        //println!("ubos: {:?}", ubos);
        Ok(UboBinding {
            binding: desc_bind.bind(),
            ubos,
            size
        })
    }
}

impl Shader {
    pub fn from_path(device: &Rc<ash::Device>, path: &String) -> Option<Shader> {
        let name = std::path::Path::new(&path).file_stem().unwrap().to_str().unwrap();
        let file_contents = utils::load_file_contents(&path)?;

        let shader_type = if path.ends_with(".frag") { vk::ShaderStageFlags::FRAGMENT } else { vk::ShaderStageFlags::COMPUTE };

        let mut shader = Self::from_contents(device, name.to_string(), shader_type, file_contents)?;

        shader.path = Some(path.clone());
        Some(shader)
    }

    pub fn from_contents(device: &Rc<ash::Device>, name: String, shader_type: vk::ShaderStageFlags, glsl_source: String) -> Option<Shader> {

        let shaderc_type = match shader_type {
            vk::ShaderStageFlags::VERTEX   => shaderc::ShaderKind::Vertex,
            vk::ShaderStageFlags::FRAGMENT => shaderc::ShaderKind::Fragment,
            _ => shaderc::ShaderKind::Compute
        };

        let spirv_artifact = Self::create_spirv(&name, shaderc_type, glsl_source)?;
        let spirv_binary : &[u32] = spirv_artifact.as_binary();

        let (stage, bindings) = match Self::reflect_descriptors(spirv_binary) {
            Ok((stage, bindings)) => (stage, bindings),
            Err(e) => {
                warnln!("err: {}", e);
                return None
            }
        };

        let module = Self::create_module(&device, spirv_binary)?;

        Some(Shader {
            device: Rc::clone(device), module, bindings,
            _name: name, path: None, stage
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

    fn create_spirv(name: &String, shader_type: shaderc::ShaderKind, glsl_source: String) -> Option<CompilationArtifact> {
        let compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_generate_debug_info();
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);

        match compiler.compile_into_spirv(&glsl_source.to_owned(),
                                          shader_type,
                                          &name,
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

    fn reflect_descriptors(binary: &[u32]) ->  Result<(vk::ShaderStageFlags, ShaderBindings)> {
        let entry_points = spirq::ReflectConfig::new()
            .spv(binary)
            .ref_all_rscs(true)
            .combine_img_samplers(true)
            .reflect()
            .unwrap();

        let entry_point = entry_points.get(0).context("No shader entry point at idx 0")?;

        use spirq::spirv::ExecutionModel::*;

        let shader_type = match entry_point.exec_model {
            Vertex => vk::ShaderStageFlags::VERTEX,
            Fragment => vk::ShaderStageFlags::FRAGMENT,
            GLCompute => vk::ShaderStageFlags::COMPUTE,
            _ => return err!("Unsupported shader type {:?}", entry_point.exec_model)
        };

        let mut images: HashMap<String, ImageBinding> = HashMap::new();
        let mut ubos: HashMap<String, UboBinding> = HashMap::new();
        let mut ssbos: HashMap<String, SsboBinding> = HashMap::new();

        for var in &entry_point.vars {
            match var {
                spirq::var::Variable::Descriptor { name, desc_bind, desc_ty, ty, nbind } => {
                    let _ = nbind;

                    let binding = desc_bind.bind();
                    let set = desc_bind.set();

                    if set != 0 {
                        return err!("Only sets at idx 0 are supported currently")
                    }

                    use spirq::ty::DescriptorType::*;

                    match desc_ty {
                        UniformBuffer() => {
                            // Getting the block name
                            let ubo_binding_name = if let Some(s) = ty.as_struct() {
                                s.clone().name.context("Ubo struct is missing a name")?
                            }
                            else { return err!("Ubo is not a struct/block type")? };

                            // Not all Ubos have an instance name, we use the instance
                            // name as a prefix if it exists, otherwise nothing
                            let ubo_prefix = name.clone().unwrap_or("".to_string());

                            ubos.insert(ubo_binding_name, UboBinding::new(ubo_prefix, desc_bind, ty)?);
                        },
                        CombinedImageSampler() => {
                            let name = name.as_ref().context("Descriptor has no name")?;
                            images.insert(name.clone(), ImageBinding{ binding, image_type: DescType::CombinedImageSampler });
                        },
                        StorageImage(_) => {
                            let name = name.as_ref().context("Descriptor has no name")?;
                            images.insert(name.clone(), ImageBinding{ binding, image_type: DescType::StorageImage });
                        },
                        StorageBuffer(_) => {
                            let size = match ty.min_nbyte() {
                                Some(bytes) => bytes, None => return err!("Ssbo has unknown byte size")
                            };

                            let ssbo_name = if let Some(s) = ty.as_struct() {
                                s.clone().name.context("Ssbo struct is missing a name")?
                            }
                            else { return err!("Ssbo is not a struct/block type")? };

                            println!("ssbo name: {:?}", ty);


                            ssbos.insert(ssbo_name, SsboBinding{ binding, size });
                        },
                        _ => println!("Note unsupported descriptor ignored {:?}", desc_ty)
                    }
                },
                _ => println!("Note unsupported descriptor ignored {:?}", var)
            };
        }

        Ok((shader_type, ShaderBindings {
            images, ubos, ssbos
        }))
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
        self.device.destroy_shader_module(self.module, None);
        }
    }
}
