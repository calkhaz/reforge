use ash::vk;

pub struct Shader {
    pub module: vk::ShaderModule
}

impl Shader {
    pub fn new(device: &ash::Device, path: &str) -> Option<Shader> {

        let glsl_source = std::fs::read_to_string(path)
            .expect("Should have been able to read the file");

        let compiler = shaderc::Compiler::new().unwrap();
        let options = shaderc::CompileOptions::new().unwrap();

        let compilation = match compiler.compile_into_spirv(&glsl_source.to_owned(),
                                                        shaderc::ShaderKind::Compute,
                                                        path,
                                                        "main",
                                                        Some(&options)) {
            Ok(binary) => {
                assert_eq!(Some(&0x07230203), binary.as_binary().first());

                Some(binary)
            } ,
            Err(e) => { eprintln!("{:?}", e); None }
        }?;

        let spirv_binary = compilation.as_binary();

        let shader_info = vk::ShaderModuleCreateInfo::builder().code(spirv_binary);

        unsafe {
        let module = match device.create_shader_module(&shader_info, None) {
            Ok(module) => Some(module),
            Err(e) => { eprintln!("{:?}", e); None }
        }?;

        Some(Shader {
            module: module
        })

        }
    }
}
