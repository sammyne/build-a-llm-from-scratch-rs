use anyhow::Context;
use burn::backend::NdArray;
use burn::prelude::*;
use chapter06::dataset::{LoadCsvOptions, SpamDataset};
use tiktoken::ext::Encoding;

type B = NdArray;
type Device = <B as Backend>::Device;

fn main() -> anyhow::Result<()> {
    let tokenizer = Encoding::gpt2();

    let device = &Device::Cpu;

    let opts = LoadCsvOptions::new("train.csv", &tokenizer, device);
    let train_dataset = SpamDataset::<B>::load_csv(opts).context("load train dataset")?;

    // 由于随机种子和算法不一致，因此和书上的有差异。
    println!("max-length(train-dataset): {}", train_dataset.max_length);

    Ok(())
}
