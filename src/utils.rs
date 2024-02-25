use std::collections::HashMap;

use crate::vulkan::pipeline::Pipeline;

use std::rc::Rc;
use std::cell::RefCell;

pub const TERM_CLEAR : &str = "\r\x1b[2K";
pub const TERM_RED   : &str = "\x1b[31m";
pub const TERM_YELLOW: &str = "\x1b[33m";
const MOVING_AVG_SIZE: f64 = 60.0;

#[macro_export]
macro_rules! warnln {
    ($($arg:tt)*) => {{
        eprintln!("\r\x1b[2K\x1b[33m{}\x1b[0m", format_args!($($arg)*));
    }}
}

pub fn load_file_contents(config_path: &str) -> Option<String> {
    let contents = match std::fs::read_to_string(config_path) {
        Ok(contents) => contents,
        Err(e) => { warnln!("Error reading file '{}' : {}", config_path, e); return None }
    };

    if contents.is_empty() {
        warnln!("File was empty: {config_path}");
        return None
    }
    Some(contents)
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

pub fn moving_avg(mut avg: f64, next_value: f64) -> f64 {

    avg -= avg / MOVING_AVG_SIZE;
    avg += next_value / MOVING_AVG_SIZE;

    return avg;
}

pub fn get_elapsed_ms(inst: &std::time::Instant) -> f64{
    return (inst.elapsed().as_nanos() as f64)/1e6 as f64;
}

