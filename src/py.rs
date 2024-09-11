use std::collections::HashMap;

use pyo3::prelude::Python;
use pyo3::types::{PyList, PyString, PyModule, PyDict};
use anyhow::{Context, Result};
use tracing::warn;

use crate::render::ParamData;

fn pydict_to_paramdata(nodes: &PyDict) -> Result<HashMap<String, HashMap<String, ParamData>>> {
    let mut node_params: HashMap<String, HashMap<String, ParamData>> = HashMap::new();

    for (node_name, node) in nodes {
        let mut params: HashMap<String, ParamData> = HashMap::new();

        let node = node.extract::<&pyo3::types::PyDict>()?;

        for (param_name, value) in node {
            let param: String = param_name.extract::<&PyString>()?.to_string();

            if value.is_instance_of::<pyo3::types::PyList>() {
                let list = value.extract::<&pyo3::types::PyList>()?;
                if let Ok(vec) = list.extract::<Vec<i32>>() {
                    params.insert(param, ParamData::IntegerArray(vec));
                }
                else if let Ok(vec) = list.extract::<Vec<f32>>() {
                    params.insert(param, ParamData::FloatArray(vec));
                }
                else {
                    warn!("Invalid vector in set_buffer");
                }
            }
            else if let Ok(val) = value.extract::<bool>() {
                params.insert(param, ParamData::Boolean(val));
            }
            else if let Ok(val) = value.extract::<i32>() {
                params.insert(param, ParamData::Integer(val));
            }
            else if let Ok(val) = value.extract::<f32>() {
                params.insert(param, ParamData::Float(val));
            }
        }

        node_params.insert(node_name.extract::<&PyString>()?.to_string(), params);
    }

    Ok(node_params)
}

pub fn py_config(path: &str) -> Result<(String, HashMap<String, HashMap<String, ParamData>>)> { 
    let script = std::fs::read_to_string(path).context(format!("Failed to read file: {}", path))?;

    Python::with_gil(|py| -> Result<(String, HashMap<String, HashMap<String, ParamData>>)> {
        let py_path = std::path::Path::new(&path);
        let sys_path: &PyList = py.import("sys")?.getattr("path")?.extract::<&PyList>()?;
        sys_path.insert(0, py_path.parent().context(format!("No base directory on python config: {}", path))?)?;

        let file_name   = py_path.file_name().context(format!("No file name on python config: {}", path))?.to_str().unwrap();
        let module_name = py_path.file_stem().context(format!("No file name on python config: {}", path))?.to_str().unwrap();

        let module = PyModule::from_code(py, &script, file_name, module_name)?;
        let graph = module.getattr("graph")?.extract::<&PyString>()?.to_string();

        // Don't require "nodes", but enforce it to be correct if it is found
        let node_params = if let Ok(nodes) = module.getattr("nodes") {
            if let Ok(nodes) = nodes.extract::<&PyDict>() {
                pydict_to_paramdata(nodes)?
            }
            else { HashMap::new() }
        } else { HashMap::new() };

        Ok((graph, node_params))
    })
}
