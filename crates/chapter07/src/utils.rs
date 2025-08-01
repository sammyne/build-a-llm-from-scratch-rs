use std::fmt::Display;
use std::fs::File;
use std::path::Path;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Data {
    pub instruction: String,
    pub input: Option<String>,
    pub output: String,
}

impl Display for Data {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string(self).expect("serde-json::to_string");
        write!(f, "{json}")
    }
}

pub fn load_json<P>(file_path: P) -> anyhow::Result<Vec<Data>>
where
    P: AsRef<Path>,
{
    let f = File::open(file_path.as_ref()).context("open file")?;

    let out = serde_json::from_reader(f).context("json decode")?;

    Ok(out)
}
