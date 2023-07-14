extern crate lalrpop_util;

use std::collections::HashMap;

use crate::vulkan::pipeline_graph::{SWAPCHAIN_OUTPUT, FILE_INPUT};

 // Synthesized by LALRPOP
lalrpop_mod!(pub config_gramar, "/config/config_grammar.rs");

pub struct ConfigDescriptor {
    pub resource_name  : String,
    pub descriptor_name: String
}

#[derive(Default)]
pub struct ConfigPipeline {
    pub shader_path: String,
    // images
    pub input_images: Vec<ConfigDescriptor>,
    pub output_images: Vec<ConfigDescriptor>,
    // buffers (ssbo)
    pub input_buffers: Vec<ConfigDescriptor>,
    pub output_buffers:Vec<ConfigDescriptor>
}

pub fn parse(_contents: String) -> HashMap<String, ConfigPipeline> {

    let config_data = HashMap::from([
        ("passthrough-pipeline".to_string(), ConfigPipeline {
            shader_path: "shaders/passthrough.comp".to_string(),
            input_images:  vec![ConfigDescriptor{resource_name: FILE_INPUT.to_string(),       descriptor_name: "input_image".to_string()}],
            output_images: vec![ConfigDescriptor{resource_name: SWAPCHAIN_OUTPUT.to_string(), descriptor_name: "output_image".to_string()}],
            //output_buffers: [("outputBuffer".to_string(), "passthrough_buffer".to_string())].to_vec(),
            ..Default::default()
        })
    ]);

    config_data
}
