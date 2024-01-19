use std::collections::HashMap;

use crate::vulkan::pipeline_graph::{FINAL_OUTPUT, FILE_INPUT};

use crate::warnln;

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
}

pub struct Config {
    pub graph_pipelines: HashMap<String, GraphPipeline>,
    pub pipeline_instances: HashMap<String, PipelineInstance>
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

        let base_path = format!("{shader_path}/{pipeline_type}");
        let path = std::path::Path::new(&base_path);

        if path.exists() {
            pipeline.file_path = base_path;
        }
        else {
            pipeline.file_path = format!("{base_path}.comp");
        }

    }

    config
}

pub fn parse(graph: String, shader_dir: &String) -> Option<Config> {
    let mut config = Config {
        graph_pipelines: HashMap::new(),
        pipeline_instances: HashMap::new()
    };

    let delimiter = "->";

    // For each graph config line - Ex: input -> filter0 -> filter1 -> output
    for line in graph.lines() {
        let mut start = 0;
        let mut graph: Vec<&str> = Vec::new();

        while let Some(next) = line[start..].find(delimiter) {
            let part = &line[start..start + next];
            graph.push(part);
            start += next + delimiter.len();
        }

        let part = &line[start..];

        graph.push(part.trim());

        // For each pipeline + descriptor - Ex: filter0, filter1, filter2:image_name
        for i in 0..graph.len() {
            let split: Vec<&str> = graph[i].split(':').collect();
            let pipeline_name = split[0].trim().to_string();
            let descriptor_name = split.get(1);

            if pipeline_name == "input" || pipeline_name == "output" || pipeline_name.is_empty() { continue }

            let info = config.graph_pipelines.entry(pipeline_name.to_string()).or_insert(
                GraphPipeline{ ..Default::default() }
            );

            // Inputs to the current pipeline
            if i > 0 {
                let split: Vec<&str> = graph[i-1].split(':').collect();
                let (input_pipeline, input_descriptor) = (split[0].trim(), split.get(1));

                let descriptor_name = descriptor_name.unwrap_or(&"input_image").to_string();

                let resource_name = if input_pipeline == "input" { FILE_INPUT.to_string() } 
                                    else { format!("{input_pipeline}:{}", input_descriptor.unwrap_or(&"output_image").to_string()) };

                info.inputs.push(ConfigDescriptor{resource_name, descriptor_name});
            }

            // Outputs to the current pipeline
            if i+1 < graph.len() {
                let split: Vec<&str> = graph[i+1].split(':').collect();
                let output_pipeline = split[0].trim();


                let descriptor_name = descriptor_name.unwrap_or(&"output_image").to_string();

                let resource_name = if output_pipeline == "output" { FINAL_OUTPUT.to_string() }
                                    else { format!("{pipeline_name}:{descriptor_name}") };

                info.outputs.push(ConfigDescriptor{resource_name, descriptor_name});
            }
        }
    }

    config = add_file_paths(config, shader_dir);

    if config.graph_pipelines.len() == 0 { warnln!("Configuration had an empty graph");  return None }

    Some(config)
}
