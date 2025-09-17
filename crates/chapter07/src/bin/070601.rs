use anyhow::Context;
use burn::backend::{Autodiff, LibTorch};
use burn::module::Module;
use burn::prelude::Backend;
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let model = chapter06::utils::load_gpt2::<B, _>("gpt2/355M", device).context("load GPT-2")?;

    B::seed(123);
    let tokenizer = Encoding::gpt2();

    let (train_loader, _test_loader, val_loader) =
        chapter07::dataset::load_and_split("instruction-data.json", &tokenizer)
            .context("load and split data loader")?;

    let train_loss =
        chapter07::loss::calc_loss_loader(train_loader.as_ref(), &model.clone().no_grad(), 5.into(), device);
    let val_loss = chapter07::loss::calc_loss_loader(val_loader.as_ref(), &model.clone().no_grad(), 5.into(), device);

    println!("Training loss: {train_loss}");
    println!("Validation loss: {val_loss}");

    Ok(())
}
