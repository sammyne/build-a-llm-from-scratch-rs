use std::path::Path;

use anyhow::Context as _;
use burn::backend::LibTorch;
use burn::prelude::Backend;
use chapter04::GptModel;
use chapter05::gpt2;
use chapter05::utils::Tokenizer as _;
use tiktoken::ext::Encoding;

type B = LibTorch;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let data_dir = &Path::new("gpt2/gpt2/124M");
    let (settings, params) = {
        let (mut s, p) = gpt2::load_settings_and_params(&data_dir).expect("load gpt2 config");
        s.drop_rate = 0.0;
        (s, p)
    };

    let mut model = GptModel::<B>::new(&settings, device);

    gpt2::load_weights_into_gpt2(params, &mut model).context("load weights into model")?;

    let tokenizer = Encoding::gpt2();

    const TEXT1: &str = "Every effort moves you";
    let idx = tokenizer.tokenize(TEXT1).to_device(device);

    B::seed(123);

    let token_ids = chapter04::utils::generate_text_simple(&model, idx, 15, settings.context_length);
    let out = tokenizer.detokenize(token_ids).context("decode output")?;
    println!("Output text:\n{out}");

    const TEXT2: &str = std::concat!(
        "Is the following text 'spam'? Answer with 'yes' or 'no':",
        " 'You are a winner you have been specially",
        " selected to receive $1000 cash or a $2000 award.'"
    );

    let token_ids = chapter04::utils::generate_text_simple(
        &model,
        tokenizer.tokenize(TEXT2).to_device(device),
        23,
        settings.context_length,
    );
    let out = tokenizer.detokenize(token_ids).context("decode output #2")?;
    println!("Output text #2:\n{out}");

    Ok(())
}
