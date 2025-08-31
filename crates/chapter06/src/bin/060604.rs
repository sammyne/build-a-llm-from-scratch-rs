use anyhow::Context as _;
use burn::backend::{Autodiff, LibTorch};
use burn::prelude::Backend;
use chapter06::dataset::{self, DataLoaderOptions, LoadCsvOptions, SpamDataset};
use chapter06::{loss, utils};
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let model = utils::load_gpt2_for_fine_tuning("gpt2/124M", device).context("load model")?;

    let tokenizer = Encoding::gpt2();

    B::seed(123);

    let opts = LoadCsvOptions::new("train.csv", &tokenizer, device);
    let train_dataset = SpamDataset::<B>::load_csv(opts).context("load train dataset")?;

    let opts = LoadCsvOptions::new("validation.csv", &tokenizer, device);
    let validation_dataset = SpamDataset::<B>::load_csv(opts).context("load validation dataset")?;

    let opts = LoadCsvOptions::new("test.csv", &tokenizer, device);
    let test_dataset = SpamDataset::<B>::load_csv(opts).context("load test dataset")?;

    // 对于训练集，丢弃不完整的最后一批。
    let opts = DataLoaderOptions::new()
        .with_shuffle_seed(Some(456))
        .with_drop_last(true);
    let train_loader = dataset::load(train_dataset, opts);

    let validation_loader = dataset::load(validation_dataset, DataLoaderOptions::new());
    let test_loader = dataset::load(test_dataset, DataLoaderOptions::new());

    let train_accuracy = loss::calc_accuracy_loader(train_loader.as_ref(), &model, device, Some(10));
    let val_accuracy = loss::calc_accuracy_loader(validation_loader.as_ref(), &model, device, Some(10));
    let test_accuracy = loss::calc_accuracy_loader(test_loader.as_ref(), &model, device, Some(10));
    println!("Training accuracy: {:.2}", train_accuracy * 100.0);
    println!("Validation accuracy: {:.2}", val_accuracy * 100.0);
    println!("Test accuracy: {:.2}", test_accuracy * 100.0);

    Ok(())
}
