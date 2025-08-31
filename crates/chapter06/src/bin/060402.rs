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

    B::seed(123);

    const TEXT2: &str = std::concat!(
        "Is the following text 'spam'? Answer with 'yes' or 'no':",
        " 'You are a winner you have been specially",
        " selected to receive $1000 cash or a $2000 award.'"
    );

    let context_length = model.pos_emb.weight.dims()[0];
    let token_ids =
        chapter04::utils::generate_text_simple(&model, tokenizer.tokenize(TEXT2).to_device(device), 23, context_length);
    let out = tokenizer.detokenize(token_ids).context("decode output #2")?;
    println!("Output text #2:\n{out}");

    Ok(())
}
