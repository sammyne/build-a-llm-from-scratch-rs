use anyhow::Context as _;
use burn::backend::{Autodiff, LibTorch};
use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::{Tensor, s};
use chapter05::utils::Tokenizer as _;
use chapter06::utils::{self};
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let model = utils::load_gpt2_for_fine_tuning("gpt2/124M", device).context("load model")?;

    let tokenizer = Encoding::gpt2();

    let inputs: Tensor<B, 2, _> = tokenizer.tokenize("Do you have time").to_device(device);

    let outputs = model.clone().no_grad().forward(inputs);

    let dim = outputs.dims().len() - 1;

    // 不是用 softmax
    let logits = outputs.slice(s![.., -1, ..]);
    let label = logits.argmax(dim);
    println!("Class label: {}", label.into_scalar());

    Ok(())
}
