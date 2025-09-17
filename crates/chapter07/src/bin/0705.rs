use anyhow::Context as _;
use burn::backend::{Autodiff, LibTorch};
use burn::prelude::Backend;
use chapter05::utils::{GenerateOptions, Tokenizer};
use chapter07::utils;
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let model = chapter06::utils::load_gpt2::<B, _>("gpt2/355M", device).context("load GPT-2")?;

    B::seed(123);
    let tokenizer = Encoding::gpt2();

    let (.., val_data) = utils::load_and_split_data("instruction-data.json").context("load and split data")?;

    let input_text = utils::format_input(&val_data[0]);
    println!("Input text: {input_text}");

    let idx = tokenizer.tokenize(&input_text).to_device(device);

    let opts =
        GenerateOptions::new(35, model.pos_emb.weight.dims()[0]).with_eos_id(Some(chapter07::PAD_TOKEN_ID as usize));
    let token_ids = chapter05::utils::generate(&model, idx, opts);

    let generated_text = tokenizer.detokenize(token_ids).context("decode output")?;
    let response_text = generated_text.split_at(input_text.len()).1.trim();
    println!("\n\nResponse text:\n{response_text}");

    Ok(())
}
