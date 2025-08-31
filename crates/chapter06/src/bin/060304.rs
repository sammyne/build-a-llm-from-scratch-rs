use anyhow::Context;
use burn::backend::NdArray;
use burn::prelude::*;
use chapter06::dataset::{self, DataLoaderOptions, LoadCsvOptions, SpamDataset};
use tiktoken::ext::Encoding;

type B = NdArray;
type Device = <B as Backend>::Device;

fn main() -> anyhow::Result<()> {
    let tokenizer = Encoding::gpt2();
    let device = &Device::Cpu;

    let opts = LoadCsvOptions::new("train.csv", &tokenizer, device);
    let train_dataset = SpamDataset::<B>::load_csv(opts).context("load train dataset")?;

    let opts = LoadCsvOptions::new("validation.csv", &tokenizer, device);
    let validation_dataset = SpamDataset::<B>::load_csv(opts).context("load validation dataset")?;

    let opts = LoadCsvOptions::new("test.csv", &tokenizer, device);
    let test_dataset = SpamDataset::<B>::load_csv(opts).context("load test dataset")?;

    B::seed(123);

    // 对于训练集，丢弃不完整的最后一批。
    let opts = DataLoaderOptions::new()
        .with_shuffle_seed(Some(456))
        .with_drop_last(true);
    let train_loader = dataset::load(train_dataset, opts);
    let validation_loader = dataset::load(validation_dataset, DataLoaderOptions::new());
    let test_loader = dataset::load(test_dataset, DataLoaderOptions::new());

    println!("{} training batches", train_loader.iter().count());
    println!("{} validation batches", validation_loader.iter().count());
    println!("{} test batches", test_loader.iter().count());

    Ok(())
}
