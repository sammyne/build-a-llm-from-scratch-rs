use anyhow::Context as _;
use burn::backend::{Autodiff, LibTorch};
use burn::module::Module;
use burn::nn::LinearConfig;
use burn::prelude::Backend;
use chapter06::dataset::{self, DataLoaderOptions, LoadCsvOptions, SpamDataset};
use chapter06::utils::RequireGradMapper;
use chapter06::{loss, utils};
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let model = utils::load_gpt2("gpt2/124M", device).context("load model")?;

    B::seed(123);

    let mut model = model.no_grad();

    const NUM_CLASSES: usize = 2;
    let emb_dim = model.tok_emb.weight.dims()[1];
    model.out_head = LinearConfig::new(emb_dim, NUM_CLASSES).with_bias(true).init(device);

    let trf_block = model.trf_blocks.last_mut().context("miss last transfomer block")?;
    *trf_block = trf_block.clone().map(&mut RequireGradMapper);

    model.final_norm = model.final_norm.clone().map(&mut RequireGradMapper);

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

    let train_loss = loss::calc_loss_loader(train_loader.as_ref(), model.clone(), device, None);
    let val_loss = loss::calc_loss_loader(validation_loader.as_ref(), model.clone(), device, None);
    let test_loss = loss::calc_loss_loader(test_loader.as_ref(), model.clone(), device, None);
    println!("Training loss: {train_loss:.3}");
    println!("Validation loss: {val_loss:.3}");
    println!("Test loss: {test_loss:.3}");

    Ok(())
}
