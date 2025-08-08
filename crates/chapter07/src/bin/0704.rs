use anyhow::Context;
use burn::backend::LibTorch;
use burn::prelude::Backend;
use chapter07::dataset::{self, DataLoaderOptions, InstructionDataset};
use chapter07::utils;
use tiktoken::ext::Encoding;

type B = LibTorch;

type Device = <B as Backend>::Device;

fn main() -> anyhow::Result<()> {
    const FILE_PATH: &str = "instruction-data.json";

    let (train_data, test_data, val_data) = utils::load_and_split_data(FILE_PATH).context("load and split")?;

    let tokenizer = Encoding::gpt2();

    const BATCH_SIZE: usize = 8;
    B::seed(123);

    let opts = {
        let customized_collate_fn = |batch: &[Vec<u32>], device: &Device| {
            utils::custom_collate_fn::<B, _>(batch, None, None, Some(1024), device)
        };
        DataLoaderOptions {
            batch_size: BATCH_SIZE,
            shuffle_seed: Some(20250808),
            num_workers: 0,
            drop_last: true,
            collate_fn: customized_collate_fn,
        }
    };

    let train_dataset = InstructionDataset::new(&train_data, &tokenizer).context("build train dataset")?;
    let train_loader = dataset::load(train_dataset, &opts);

    let test_dataset = InstructionDataset::new(&test_data, &tokenizer).context("build test dataset")?;
    let _test_loader = dataset::load(test_dataset, &opts);

    let val_dataset = InstructionDataset::new(&val_data, &tokenizer).context("build val dataset")?;
    let _val_loader = dataset::load(val_dataset, &opts);

    println!("Train Loader:");
    for (inputs, targets) in train_loader.iter() {
        println!("{:?}, {:?}", inputs.shape(), targets.shape());
    }

    Ok(())
}
