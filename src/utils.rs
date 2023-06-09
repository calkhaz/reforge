use std::collections::HashMap;

use crate::vulkan::pipeline_graph::PipelineInfo;

const MOVING_AVG_SIZE: f64 = 60.0;

pub fn get_modified_times(pipeline_infos: &HashMap<&str, PipelineInfo>) -> HashMap<String, u64> {
    let mut timestamps: HashMap<String, u64> = HashMap::new();

    for (name, info) in pipeline_infos {
        match std::fs::metadata(&info.shader_path) {
            Ok(metadata) => {
                let timestamp = metadata.modified().unwrap().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_secs();
                timestamps.insert(name.to_string(), timestamp);
            },
            // Set the modification time to zero so it gets picked up
            // up when the file is findable again
            Err(_) => {
                timestamps.insert(name.to_string(), 0);
            }
        };
    }

    timestamps
}

pub fn get_dim(width: u32, height: u32, new_width: Option<u32>, new_height: Option<u32>) -> (u32, u32) {
    let mut w  = width;
    let mut h = height;

    if new_width.is_some() && new_height.is_some() {
        return (new_width.unwrap(), new_height.unwrap())
    }

    if new_width.is_some() {
        w = new_width.unwrap();
        h = ((w as f32/width as f32)*height as f32) as u32;
    }
    else if new_height.is_some() {
        h = new_height.unwrap();
        w = ((h as f32/(height as f32))*width as f32) as u32;
    }

    (w, h)
}

pub fn moving_avg(mut avg: f64, next_value: f64) -> f64 {

    avg -= avg / MOVING_AVG_SIZE;
    avg += next_value / MOVING_AVG_SIZE;

    return avg;
}

pub fn get_elapsed_ms(inst: &std::time::Instant) -> f64{
    return (inst.elapsed().as_nanos() as f64)/1e6 as f64;
}

