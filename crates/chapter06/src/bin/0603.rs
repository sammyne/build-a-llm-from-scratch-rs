use std::collections::HashSet;

use anyhow::Context;
use burn::backend::NdArray;
use burn::prelude::*;
use chapter06::dataset::{self, DataLoaderOptions, LoadCsvOptions, SpamDataset};
use tiktoken::ext::Encoding;

type B = NdArray;
type Device = <B as Backend>::Device;

fn main() -> anyhow::Result<()> {
    let tokenizer = Encoding::gpt2();

    let allowed_specials = ["<|endoftext|>"].into_iter().collect::<HashSet<_>>();
    println!("encoded: {:?}", tokenizer.encode("<|endoftext|>", &allowed_specials));

    let device = &Device::Cpu;

    let opts = LoadCsvOptions::new("train.csv", &tokenizer, device);
    let train_dataset = SpamDataset::<B>::load_csv(opts).context("load train dataset")?;
    // 由于随机种子和算法不一致，因此和书上的有差异。
    println!("max-length(train-dataset): {}", train_dataset.max_length);

    let opts = LoadCsvOptions::new("validation.csv", &tokenizer, device);
    let validation_dataset = SpamDataset::<B>::load_csv(opts).context("load validation dataset")?;

    let opts = LoadCsvOptions::new("test.csv", &tokenizer, device);
    let test_dataset = SpamDataset::<B>::load_csv(opts).context("load test dataset")?;

    B::seed(123);

    // 对于训练集，丢弃不完整的最后一批。
    let train_loader = dataset::load(train_dataset, DataLoaderOptions::default().with_drop_last(true));
    let validation_loader = dataset::load(validation_dataset, Default::default());
    let test_loader = dataset::load(test_dataset, Default::default());

    for (x, y) in train_loader.iter() {
        println!("Input batch dimensions: {:?}", x.shape());
        println!("Label batch dimensions: {:?}", y.shape());
        break
    }

    println!("{} training batches", train_loader.iter().count());
    println!("{} validation batches", validation_loader.iter().count());
    println!("{} test batches", test_loader.iter().count());

    Ok(())
}
