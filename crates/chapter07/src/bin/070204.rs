use anyhow::Context;
use chapter07::utils::{self, Data};

fn main() -> anyhow::Result<()> {
    const FILE_PATH: &str = "instruction-data.json";

    let data: Vec<Data> = utils::load_json(FILE_PATH).context("load")?;

    let model_input = utils::format_input(&data[999]);
    let desired_response = format!("\n\n### Response:\n{}", data[999].output);
    println!("{model_input}{desired_response}");

    Ok(())
}
