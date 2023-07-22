extern crate lalrpop_util;

use std::collections::HashMap;

use crate::config::config::config_gramar::ExprListParser;
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
pub struct GraphPipeline {
    // images and ssbos
    pub inputs : Vec<ConfigDescriptor>,
    pub outputs: Vec<ConfigDescriptor>,
}

pub struct PipelineInstance {
    pub pipeline_type: String
}

pub struct Config {
    pub graph_pipelines: HashMap<String, GraphPipeline>,
    pub pipeline_instances: HashMap<String, PipelineInstance>
}

pub fn parse(contents: String) -> Option<Config> {

    let ast_exprs: Vec<Box<ast::Expr>> =
        match ExprListParser::new().parse(&contents) {
            Ok(ast) => Some(ast),
            Err(err) => { eprintln!("Failed to parse the input: {}", err); None }
        }?;

    let mut config = Config {
        graph_pipelines: HashMap::new(),
        pipeline_instances: HashMap::new()
    };

    let mut found_input  = false;
    let mut found_output = false;

    for expr in ast_exprs {
        match *expr {
            ast::Expr::Graph(graph) => {
                for i in 0..graph.len() {
                    let pipeline = &graph[i];
                    let pipeline_name = &pipeline.0;
                    let pipeline_descriptor_name = &pipeline.1;

                    if pipeline_name == "input"  { found_input  = true; continue; }
                    if pipeline_name == "output" { found_output = true; continue; }

                    let info = config.graph_pipelines.entry(pipeline_name.to_string()).or_insert(
                        GraphPipeline{
                            ..Default::default()
                        }
                    );

                    // Inputs
                    if i > 0 {
                        let input_pipeline = &graph[i-1];
                        let input_pipeline_name = &input_pipeline.0;

                        let descriptor_name = pipeline_descriptor_name.clone().unwrap_or("input_image".to_string());
                        let input_descriptor = input_pipeline.1.clone();

                        let resource_name = if input_pipeline_name == "input" { FILE_INPUT.to_string() } 
                                            else { format!("{input_pipeline_name}:{}", input_descriptor.unwrap_or("output_image".to_string())) };

                        info.inputs.push(ConfigDescriptor{resource_name, descriptor_name});
                    }

                    // Outputs
                    if i+1 < graph.len() {
                        let output_pipeline = &graph[i+1];
                        let output_pipeline_name = &output_pipeline.0;

                        let descriptor_name = pipeline_descriptor_name.clone().unwrap_or("output_image".to_string());

                        let resource_name = if output_pipeline_name == "output" { SWAPCHAIN_OUTPUT.to_string() }
                                            else { format!("{pipeline_name}:{descriptor_name}") };

                        info.outputs.push(ConfigDescriptor{resource_name, descriptor_name});
                    }
                }
            },
            ast::Expr::Pipeline(pipeline) => {
                config.pipeline_instances.insert(pipeline.name, PipelineInstance {pipeline_type: pipeline.pipeline_type});
            }
            _ => {}
        };
    }

    if config.graph_pipelines.len() == 0 { eprintln!("Cofiguration had an empty graph");  return None }
    if !found_input  { eprintln!("'input' is never used in the pipeline configuration");  return None }
    if !found_output { eprintln!("'output' is never used in the pipeline configuration"); return None }

    Some(config)
}
