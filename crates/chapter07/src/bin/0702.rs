use std::fs;
use std::path::Path;

use anyhow::Context;
use chapter07::utils::{self, Data};

fn main() -> anyhow::Result<()> {
    const FILE_PATH: &str = "instruction-data.json";
    const URL: &str = std::concat!(
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch",
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    );

    let data = download_and_load_file(FILE_PATH, URL)?;
    println!("Number of entries: {}", data.len());

    println!("Example entry: {}\n", data[50]);
    println!("Another example entry: {}\n", data[999]);

    let model_input = utils::format_input(&data[50]);
    let desired_response = format!("\n\n### Response:\n{}", data[50].output);
    println!("{model_input}{desired_response}");

    let model_input = utils::format_input(&data[999]);
    let desired_response = format!("\n\n### Response:\n{}", data[999].output);
    println!("\n\n{model_input}{desired_response}");

    // Listing 7.3 Partitioning the dataset
    let train_portion = (data.len() as f32 * 0.85) as usize;
    let test_portion = (data.len() as f32 * 0.1) as usize;
    // let val_portion = data.len() - train_portion - test_portion;

    let train_data = &data[..train_portion];
    let test_data = &data[train_portion..][..test_portion];
    let val_data = &data[(train_portion + test_portion)..];

    println!("\n");
    println!("Training set length: {}", train_data.len());
    println!("Validation set length: {}", val_data.len());
    println!("Test set length: {}", test_data.len());

    Ok(())
}

/// Listing 7.1 Downloading the dataset
fn download_and_load_file<P>(file_path: P, url: &str) -> anyhow::Result<Vec<Data>>
where
    P: AsRef<Path>,
{
    let path = file_path.as_ref();
    if !path.exists() {
        let r = reqwest::blocking::get(url).context("http get")?;
        if !r.status().is_success() {
            anyhow::bail!("bad http status: {}", r.status());
        }

        let body = r.text().context("read http response body")?;
        fs::write(path, body).context("save the downloaded file")?;
    }

    utils::load_json(path).context("load")
}
