use std::fmt::Debug;
use std::collections::HashMap;

#[derive(Debug)]
pub enum ParamValue {
    Number(i32),
    //Float(f32),
    //Bool(bool),
    //Error,
}

#[derive(Debug)]
pub enum NodeExpr {
    // (node-name, node-type, node-parameters)
    Node((String, String, HashMap<String, Box<ParamValue>>)),
    // vec[(node-name, param-name) -> (node-name2, param-name2) -> ...]
    Graph(Vec<(String, Option<String>)>),
    Ignore(i32)
}
