use std::collections::{BTreeMap, HashMap};

use pyo3::prelude::Python;
use pyo3::types::{PyAny, PyList, PyString, PyModule, PyDict};
use anyhow::{anyhow, Context, Result};
use tracing::warn;

use crate::utils;
use crate::err;
use crate::render::ParamData;
use crate::ui::UiParam;

pub struct PyConfig {
    pub graph: String,
    pub node_params: HashMap<String, HashMap<String, ParamData>>,
    pub ui_params: BTreeMap<String, UiParam>, // BTreeMap so the UI nodes are sorted consistently
    pub module: pyo3::Py<PyModule>,
    pub timestamp: u64,
    pub path: String
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
fn ui_to_paramdata(nodes: &PyDict) -> Result<BTreeMap<String, UiParam>> {
    let mut node_params: BTreeMap<String, UiParam> = BTreeMap::new();

    for (node_name, ui_dict) in nodes {
        let node_name = node_name.extract::<&PyString>()?.to_string();

        let ui_dict = ui_dict.extract::<&pyo3::types::PyDict>()?;

        let val = ui_dict.get_item("val")?.context("UI entry must have 'val' entry")?;
        let min = ui_dict.get_item("min")?.context("UI entry must have 'min' entry")?;
        let max = ui_dict.get_item("max")?.context("UI entry must have 'max' entry")?;

        let val = pyany_to_param_data(val)?;
        let min = pyany_to_param_data(min)?;
        let max = pyany_to_param_data(max)?;

        node_params.insert(node_name, UiParam{ val, min, max });
    }

    Ok(node_params)
}

/* Outputting data like this
dict(
    sigma = 2.0,
    kernel_radius = 9,
) */
pub fn build_nodes_from_paramdata(module: &pyo3::Py<PyModule>, nodes: &BTreeMap<String, UiParam>) -> Result<HashMap<String, HashMap<String, ParamData>>> {
    Python::with_gil(|py| -> Result<_> {
        use ParamData::*;
        let dict = PyDict::new(py);
        for (name, node) in nodes {
            match node.val.clone() {
                Float(v) => dict.set_item(name, v)?,
                Integer(v) => dict.set_item(name, v)?,
                Boolean(v) => dict.set_item(name, v)?,
                FloatArray(v) => {
                    let py_list = PyList::new(py, &v);
                    dict.set_item(name, py_list)?;
                },
                IntegerArray(v) => {
                    let py_list = PyList::new(py, &v);
                    dict.set_item(name, py_list)?;
                }
            }
        }

        let module = module.as_ref(py);
        let pydict_nodes = module.getattr("build_nodes")?.call1((dict,))?;

        let dict = pydict_nodes.extract::<&PyDict>()?;
        let nodes = nodes_to_paramdata(&dict)?;
        Ok(nodes)
    })
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
        let ui_nodes = module.getattr("ui")?.extract::<&PyDict>()?;

        let ui_params = ui_to_paramdata(ui_nodes)?;

        let module : pyo3::Py<PyModule> = module.into();

        let node_params = build_nodes_from_paramdata(&module, &ui_params)?;

        let timestamp = utils::get_modified_time(path);

        Ok(PyConfig{ graph, node_params, ui_params, module, timestamp, path: path.to_string()})
    })
}
