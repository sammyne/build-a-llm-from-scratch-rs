use anyhow::Context;
use chapter07::utils::{self, Data};

fn main() -> anyhow::Result<()> {
    const FILE_PATH: &str = "instruction-data.json";

    let data: Vec<Data> = utils::load_json(FILE_PATH).context("load")?;

    // Listing 7.3 Partitioning the dataset
    let train_portion = (data.len() as f32 * 0.85) as usize;
    let test_portion = (data.len() as f32 * 0.1) as usize;
    // let val_portion = data.len() - train_portion - test_portion;

    let train_data = &data[..train_portion];
    let test_data = &data[train_portion..][..test_portion];
    let val_data = &data[(train_portion + test_portion)..];

    println!("Training set length: {}", train_data.len());
    println!("Validation set length: {}", val_data.len());
    println!("Test set length: {}", test_data.len());

    Ok(())
}
