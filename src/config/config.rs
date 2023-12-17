extern crate lalrpop_util;

use std::collections::HashMap;

use crate::config::config::config_gramar::ExprListParser;
use crate::config::ast;

use crate::vulkan::pipeline_graph::{SWAPCHAIN_OUTPUT, FILE_INPUT};

use crate::utils::{TERM_RED, TERM_YELLOW};
use crate::warnln;

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
    pub file_path: String
}

pub struct PipelineInstance {
    pub pipeline_type: String,
    pub parameters: HashMap<String, String> // param-key -> param-value (float/bool/int in string form)
}

pub struct Config {
    pub graph_pipelines: HashMap<String, GraphPipeline>,
    pub pipeline_instances: HashMap<String, PipelineInstance>
}

// Find the line number, contents and offset in the line for a given buffer and offset
fn get_line_number_and_contents(buffer: &str, mut offset: usize) -> (usize, &str, usize) {
    let mut line_number = 1;
    let mut line_contents = "";

    for line in buffer.lines() {
        let line_length = line.len() + 1; // Add 1 for the newline character
        if offset < line_length {
            line_contents = line;
            break;
        } else {
            offset -= line_length;
            line_number += 1;
        }
    }
    (line_number, line_contents, offset)
}

// Add shader file paths to the configuration based
fn add_file_paths(mut config: Config, shader_path: &String) -> Config {
    for (pipeline_name, pipeline) in &mut config.graph_pipelines {
        // First we look for pipeline instance that has been specified and use its type
        // If that does not exist, we assume the specified name is also the type
        let pipeline_type = {
            let instance = config.pipeline_instances.get(pipeline_name);
            match instance {
                Some(inst) => inst.pipeline_type.clone(),
                None       => pipeline_name.clone()
            }
        };

        pipeline.file_path = format!("{shader_path}/{pipeline_type}.comp");
    }

    config
}

pub fn single_shader_parse(path: String, expects_input: bool) -> Config {
    // Remove path and extension to get the name
    let name = std::path::Path::new(&path).file_stem().unwrap().to_str().unwrap();

    let mut config = match expects_input {
        true  => parse(format!("input -> {name} -> output").to_string(), expects_input),
        false => parse(format!(         "{name} -> output").to_string(), expects_input)
    }.unwrap();

    // Set the file path on the parsed config
    config.graph_pipelines.get_mut(name).unwrap().file_path = path.clone();

    config
}

pub fn parse_file(contents: String, expects_input: bool, shader_path: &String) -> Option<Config> {
    let mut config = parse(contents, expects_input)?;
    config = add_file_paths(config, shader_path);
    Some(config)
}

fn parse(contents: String, expects_input: bool) -> Option<Config> {
    if contents.trim().is_empty() {
        warnln!("Empty configuration given to parse");
        return None
    }

    let ast_exprs: Vec<Box<ast::Expr>> =
        match ExprListParser::new().parse(&contents) {
            Ok(ast) => Some(ast),
            Err(lalrpop_util::ParseError::InvalidToken { location }) => {
                let (line_num, line, line_offset) = get_line_number_and_contents(&contents, location);
                let token = contents.chars().nth(location).unwrap();
                let before_token = &line[0..line_offset];
                let after_token = &line[line_offset+1..];
                warnln!("Invalid token '{token}' at line {line_num}: {before_token}{TERM_RED}{token}{TERM_YELLOW}{after_token}");
                None
            },
            Err(lalrpop_util::ParseError::UnrecognizedToken { token, expected }) => {
                let (line_num, line, line_offset) = get_line_number_and_contents(&contents, token.0);
                let (line_num2, line2, line_offset2) = get_line_number_and_contents(&contents, token.2);
                let token_str = &contents[token.0 .. token.2].trim_end_matches('\n');

                let before_token = &line[0..line_offset];
                let after_token = if line_num == line_num2 { &line2[line_offset2..] } else { "" };

                // Take the expected vector and turn it into a string with all the extra stuff trimmed
                let expected_trimmed: Vec<String> = expected.clone().iter().map(|exp| {
                    format!("'{}'", exp.trim_matches('"')
                       .trim_start_matches("r#\"")
                       .trim_end_matches("\"#"))
                } ).collect();
                let expected_str = expected_trimmed.join(", ");

                warnln!("Unrecognized token '{token_str}' at line {line_num}: {before_token}{TERM_RED}{token_str}{TERM_YELLOW}{after_token}");
                warnln!("Expected to find: {expected_str}");

                None
            },
            Err(err) => { warnln!("Error while parsing: {:?}", err); None }
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
                config.pipeline_instances.insert(pipeline.name, PipelineInstance {pipeline_type: pipeline.pipeline_type,
                                                                                  parameters: pipeline.parameters});
            }
            _ => {}
        };
    }

    if config.graph_pipelines.len() == 0 { warnln!("Cofiguration had an empty graph");  return None }
    if found_input && !expects_input { warnln!("Found 'input' in pipeline configuration but no input image was specified");  return None }
    if !found_output { warnln!("'output' is never used in the pipeline configuration"); return None }

    Some(config)
}
