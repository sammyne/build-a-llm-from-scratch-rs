use anyhow::Context;
use chapter07::utils::{self, Data};

fn main() -> anyhow::Result<()> {
    const FILE_PATH: &str = "instruction-data.json";

    let data: Vec<Data> = utils::load_json(FILE_PATH).context("load")?;
    println!("Number of entries: {}", data.len());

    Ok(())
}
