use std::fmt::Debug;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Pipeline {
    pub name: String,
    pub pipeline_type: String,
    pub parameters: HashMap<String, String>
}

#[derive(Debug)]
pub enum Expr {
    Pipeline(Pipeline),
    // vec[(pipeline-name, param-name) -> (pipeline-name2, param-name2) -> ...]
    Graph(Vec<(String, Option<String>)>),
    Ignore(i32)
}
