use std::path::Path;

use anyhow::Context as _;
use burn::backend::{Autodiff, LibTorch};
use burn::prelude::Backend;
use chapter04::GptModel;
use chapter05::gpt2;
use chapter05::utils::Tokenizer;
use chapter07::utils;
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cuda(0);

    let data_dir = &Path::new("gpt2/gpt2/355M");
    let (settings, params) = {
        let (mut s, p) = gpt2::load_settings_and_params(&data_dir).expect("load gpt2 config");
        s.drop_rate = 0.0;
        (s, p)
    };

    let mut model = GptModel::<B>::new(&settings, device);

    gpt2::load_weights_into_gpt2(params, &mut model).context("load weights into model")?;

    B::seed(123);
    let tokenizer = Encoding::gpt2();

    let (.., val_data) = utils::load_and_split_data("instruction-data.json").context("load and split data")?;

    let input_text = utils::format_input(&val_data[0]);
    println!("Input text: {input_text}");

    let idx = tokenizer.tokenize(&input_text).to_device(device);
    let token_ids = chapter05::utils::generate(
        &model,
        idx,
        35,
        settings.context_length,
        None,
        None,
        Some(chapter07::PAD_TOKEN_ID as usize),
    );

    let generated_text = tokenizer.detokenize(token_ids).context("decode output")?;
    let response_text = generated_text.split_at(input_text.len()).1.trim();
    println!("\n\nResponse text:\n{response_text}");

    Ok(())
}
