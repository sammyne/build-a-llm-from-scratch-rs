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

    B::seed(123);

    // 对于训练集，丢弃不完整的最后一批。
    let opts = DataLoaderOptions::new()
        .with_shuffle_seed(Some(456))
        .with_drop_last(true);
    let train_loader = dataset::load(train_dataset, opts);

    for (x, y) in train_loader.iter() {
        println!("Input batch dimensions: {:?}", x.shape());
        println!("Label batch dimensions: {:?}", y.shape());
        break;
    }

    Ok(())
}
