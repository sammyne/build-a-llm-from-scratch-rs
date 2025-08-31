use std::path::Path;

use anyhow::Context as _;
use burn::backend::{Autodiff, LibTorch};
use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use chapter04::GPT_124M;
use chapter06::dataset::{self, DataLoaderOptions, LoadCsvOptions, SpamDataset};
use chapter06::loss;
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;

type Device = <LibTorch as Backend>::Device;

/// 依赖 060701 训练出的模型。
fn main() -> anyhow::Result<()> {
    let model_path = Path::new("spam-classifier-model.mpk");
    if !model_path.exists() {
        anyhow::bail!("model not exists");
    }

    let device = &Device::Cpu;

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    // TODO：查明 with_qkv_bias 的具体作用。实验显示加了这个配置才能还原出和 060701 训练出的模型效果。
    let model = GPT_124M
        .with_qkv_bias(true)
        .init::<B>(device)
        .load_file("spam-classifier-model.mpk", &recorder, device)
        .context("load model")?;
    let model = model.no_grad();

    let tokenizer = Encoding::gpt2();

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

    let train_accuracy = loss::calc_accuracy_loader(train_loader.as_ref(), &model, device, None);
    let val_accuracy = loss::calc_accuracy_loader(validation_loader.as_ref(), &model, device, None);
    let test_accuracy = loss::calc_accuracy_loader(test_loader.as_ref(), &model, device, None);

    println!("Train accuracy: {:.2}%", train_accuracy * 100.0);
    println!("Validation accuracy: {:.2}%", val_accuracy * 100.0);
    println!("Test accuracy: {:.2}%", test_accuracy * 100.0);

    Ok(())
}
