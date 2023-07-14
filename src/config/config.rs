extern crate lalrpop_util;

use std::collections::HashMap;

use crate::config::config::config_gramar::NodeExprListParser;
use crate::config::ast;

use crate::vulkan::pipeline_graph::{SWAPCHAIN_OUTPUT, FILE_INPUT};

 // Synthesized by LALRPOP
lalrpop_mod!(pub config_gramar, "/config/config_grammar.rs");

#[derive(Debug)]
pub struct ConfigDescriptor {
    pub resource_name  : String,
    pub descriptor_name: String
}

#[derive(Default, Debug)]
pub struct ConfigPipeline {
    pub shader_path: String,
    // images
    pub input_images: Vec<ConfigDescriptor>,
    pub output_images: Vec<ConfigDescriptor>,
    // buffers (ssbo)
    pub input_buffers: Vec<ConfigDescriptor>,
    pub output_buffers:Vec<ConfigDescriptor>
}

pub fn parse(contents: String) -> HashMap<String, ConfigPipeline> {

    let ast_exprs: Vec<Box<ast::NodeExpr>> = NodeExprListParser::new()
        .parse(&contents)
        .expect("Failed to parse the input");

    let mut config_data: HashMap<String, ConfigPipeline> = HashMap::new();

    for expr in ast_exprs {
        match *expr {
            ast::NodeExpr::Graph(graph) => {
                // change this to get the -1 and +1 values for input/output
                for i in 0..graph.len() {
                    let node = &graph[i];
                    let node_name = &node.0;
                    let node_descriptor_name = &node.1;

                    if node_name == "input" || node_name == "output" {
                        continue;
                    }

                    let info = config_data.entry(node_name.to_string()).or_insert(
                        ConfigPipeline{
                            shader_path: format!("shaders/{node_name}.comp"),
                            ..Default::default()
                        }
                    );

                    // Input images
                    if i > 0 {
                        let input_node = &graph[i-1];
                        let input_node_name = &input_node.0;

                        let descriptor_name = node_descriptor_name.clone().unwrap_or("input_image".to_string());
                        let input_descriptor = input_node.1.clone();

                        let resource_name = if input_node_name == "input" { FILE_INPUT.to_string() } 
                                            else { format!("{input_node_name}:{}", input_descriptor.unwrap_or("output_image".to_string())) };

                        info.input_images.push(ConfigDescriptor{resource_name, descriptor_name});
                    }

                    // Output images
                    if i+1 < graph.len() {
                        let output_node = &graph[i+1];
                        let output_node_name = &output_node.0;

                        let descriptor_name = node_descriptor_name.clone().unwrap_or("output_image".to_string());

                        let resource_name = if output_node_name == "output" { SWAPCHAIN_OUTPUT.to_string() }
                                            else { format!("{node_name}:{descriptor_name}") };

                        info.output_images.push(ConfigDescriptor{resource_name, descriptor_name});
                    }
                }
            },
            _ => {}
        };
    }

    config_data
}
