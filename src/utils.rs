use std::collections::HashMap;

use crate::vulkan::pipeline::Pipeline;

use std::rc::Rc;
use std::cell::RefCell;
use tracing::{debug, trace};
use anyhow::{anyhow, Result, Context};

pub const TERM_CLEAR : &str = "\r\x1b[2K";

#[macro_export]
macro_rules! err {
    ($($arg:tt)*) => {{
        Err(anyhow!($($arg)*))
    }};
}

pub fn load_file_contents(config_path: &str) -> Result<String> {
    let contents = std::fs::read_to_string(config_path).context(format!("Reading {}", config_path))?;

    if contents.is_empty() {
        return err!("File was empty: {config_path}");
    }

    Ok(contents)
}

pub fn get_modified_time(path: &String) -> u64 {
    match std::fs::metadata(path) {
        Ok(metadata) => {
            metadata.modified().unwrap().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_secs()
        },
        // Set the modification time to zero so it gets picked up
        // up when the file is findable again
        Err(_) => 0
    }
}

pub fn get_modified_times(pipelines: &HashMap<String, Rc<RefCell<Pipeline>>>) -> HashMap<String, u64> {
    let mut timestamps: HashMap<String, u64> = HashMap::new();

    for (name, pipeline) in pipelines {
        if let Some(path) = pipeline.borrow().info.shader.borrow().path.as_ref() {
            timestamps.insert(name.to_string(), get_modified_time(&path));
        }
    }

    timestamps
}

pub fn get_dim(width: u32, height: u32, new_width: Option<u32>, new_height: Option<u32>) -> (u32, u32) {
    match (new_width, new_height) {
        (Some(new_width), Some(new_height)) => (new_width, new_height),
        (Some(new_width), None)             => (new_width, ((new_width as f32/width as f32)*height as f32) as u32),
        (None,            Some(new_height)) => (((new_height as f32/(height as f32))*width as f32) as u32, new_height),
        (None,            None)             => (width, height)
    }
}

/*
pub fn moving_avg(mut avg: f64, next_value: f64) -> f64 {

    const MOVING_AVG_SIZE: f64 = 60.0;
    avg -= avg / MOVING_AVG_SIZE;
    avg += next_value / MOVING_AVG_SIZE;

    return avg;
}

pub fn get_elapsed_ms(inst: &std::time::Instant) -> f64{
    return (inst.elapsed().as_nanos() as f64)/1e6 as f64;
}
*/

fn file_exists(path: &str) -> bool {
    std::path::Path::new(path).is_file()
}

pub fn find_python_config(python_config: &str, mut python_path: Option<String>) -> Result<String> {
    if python_path.is_none() {
        if let Ok(env_path) = std::env::var("REFORGE_PY_CONFIG_PATH") {
            trace!("Setting py_config_path via env to: {}", env_path);
            python_path = Some(env_path);
        }
    }

    debug!("Looking for {python_config} in {:?}", python_path);

    let inferred_path = if let Some(python_path) = python_path.as_ref() {
        let file_path = format!("{}/{}", python_path, python_config);
        let file_path_py = format!("{}.py", file_path);
        println!("{} {}", file_path, file_path_py);

        if file_exists(&file_path)        { Some(file_path) }
        else if file_exists(&file_path_py){ Some(file_path_py) }
        else { None }
    }
    else { None };

    let path = if let Some(p) = inferred_path.as_ref() { &p } else { python_config };

    if file_exists(path) {
        Ok(path.to_string())
    }
    else if let Some(python_path) = python_path {
        Err(anyhow!("Could not find python config: {} in {}", python_config, python_path))
    }
    else {
        Err(anyhow!("Could not find python config: {}", python_config))
    }
}
