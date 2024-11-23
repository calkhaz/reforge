use std::collections::HashMap;

use pyo3::prelude::Python;
use pyo3::types::{PyAny, PyList, PyString, PyModule, PyDict};
use anyhow::{anyhow, Context, Result};
use tracing::warn;

use crate::err;
use crate::render::ParamData;
use crate::ui::UiParam;

pub struct PyConfig {
    pub graph: String,
    pub node_params: HashMap<String, HashMap<String, ParamData>>,
    pub ui_params: HashMap<String, UiParam>
}

fn pyany_to_param_data(value: &PyAny) -> Result<ParamData> {
    let pd = if value.is_instance_of::<pyo3::types::PyList>() {
        let list = value.extract::<&pyo3::types::PyList>()?;
        if let Ok(vec) = list.extract::<Vec<i32>>() {
            ParamData::IntegerArray(vec)
        }
        else if let Ok(vec) = list.extract::<Vec<f32>>() {
            ParamData::FloatArray(vec)
        }
        else {
            return err!("Invalid vector in set_buffer");
        }
    }
    else if let Ok(val) = value.extract::<bool>() {
        ParamData::Boolean(val)
    }
    else if let Ok(val) = value.extract::<i32>() {
        ParamData::Integer(val)
    }
    else if let Ok(val) = value.extract::<f32>() {
        ParamData::Float(val)
    }
    else {
        return err!("Unable to convert value of {:?}", value);
    };

    Ok(pd)
}


/* Receive data like this from python
ui = dict(
    sigma         = dict(val = 2.0, min = 1.0,  max = 50),
    kernel_radius = dict(val = 9,   min = 1,    max = 30),
)
*/
fn ui_to_paramdata(nodes: &PyDict) -> Result<HashMap<String, UiParam>> {
    let mut node_params: HashMap<String, UiParam> = HashMap::new();

    for (node_name, ui_dict) in nodes {
        let node_name = node_name.extract::<&PyString>()?.to_string();

        let ui_dict = ui_dict.extract::<&pyo3::types::PyDict>()?;

        let val = ui_dict.get_item("val")?.context("UI entry must have 'min' entry")?;
        let min = ui_dict.get_item("min")?.context("UI entry must have 'min' entry")?;
        let max = ui_dict.get_item("max")?.context("UI entry must have 'min' entry")?;

        let val = pyany_to_param_data(val)?;
        let min = pyany_to_param_data(min)?;
        let max = pyany_to_param_data(max)?;

        node_params.insert(node_name, UiParam{ val, min, max });
    }

    Ok(node_params)
}

fn nodes_to_paramdata(nodes: &PyDict) -> Result<HashMap<String, HashMap<String, ParamData>>> {
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

pub fn py_config(path: &str) -> Result<PyConfig> { 
    let script = std::fs::read_to_string(path).context(format!("Failed to read file: {}", path))?;

    Python::with_gil(|py| -> Result<PyConfig> {
        let py_path = std::path::Path::new(&path);
        let sys_path: &PyList = py.import("sys")?.getattr("path")?.extract::<&PyList>()?;
        sys_path.insert(0, py_path.parent().context(format!("No base directory on python config: {}", path))?)?;

        let file_name   = py_path.file_name().context(format!("No file name on python config: {}", path))?.to_str().unwrap();
        let module_name = py_path.file_stem().context(format!("No file name on python config: {}", path))?.to_str().unwrap();

        let module = PyModule::from_code(py, &script, file_name, module_name)?;
        let graph = module.getattr("graph")?.extract::<&PyString>()?.to_string();
        let ui_nodes = module.getattr("ui2")?.extract::<&PyDict>()?;

        let ui_params = ui_to_paramdata(ui_nodes)?;

        // Don't require "nodes", but enforce it to be correct if it is found
        let node_params = if let Ok(nodes) = module.getattr("nodes") {
            if let Ok(nodes) = nodes.extract::<&PyDict>() {
                nodes_to_paramdata(nodes)?
            }
            else { HashMap::new() }
        } else { HashMap::new() };

        Ok(PyConfig{ graph, node_params, ui_params})
    })
}
