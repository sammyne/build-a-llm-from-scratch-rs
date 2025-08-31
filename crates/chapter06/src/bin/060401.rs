use anyhow::Context as _;
use burn::backend::LibTorch;
use burn::prelude::Backend;
use chapter05::utils::Tokenizer as _;
use chapter06::utils;
use tiktoken::ext::Encoding;

type B = LibTorch;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let model = utils::load_gpt2::<B, _>("gpt2/124M", device).context("load model")?;

    let tokenizer = Encoding::gpt2();

    const TEXT1: &str = "Every effort moves you";
    let idx = tokenizer.tokenize(TEXT1).to_device(device);

    B::seed(123);

    let context_length = model.pos_emb.weight.dims()[0];
    let token_ids = chapter04::utils::generate_text_simple(&model, idx, 15, context_length);
    let out = tokenizer.detokenize(token_ids).context("decode output")?;
    println!("Output text:\n{out}");

    Ok(())
}
