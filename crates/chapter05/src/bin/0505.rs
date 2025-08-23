use std::path::Path;

use anyhow::Context as _;
use burn::backend::LibTorch;
use burn::prelude::Backend;
use chapter05::config::GPT_124M;
use chapter05::gpt2;
use chapter05::utils::{self, GenerateOptions, Tokenizer as _};
use tiktoken::ext::Encoding;

type B = LibTorch;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let data_dir = &Path::new("gpt2/124M");
    let (settings, params) = gpt2::load_settings_and_params(&data_dir).expect("load gpt2 config");

    let c = GPT_124M
        .with_context_length(settings.context_length)
        .with_qkv_bias(true);

    let mut model = c.init::<B>(device);

    gpt2::load_weights_into_gpt2(params, &mut model).context("load weights into model")?;

    let tokenizer = Encoding::gpt2();
    let idx = tokenizer.tokenize("Every effort moves you").to_device(device);

    B::seed(123);

    let opts = GenerateOptions::new(25, c.context_length)
        .with_topk(50.into())
        .with_temperature(1.5);

    let token_ids = utils::generate(&model, idx, opts);
    let out = tokenizer.detokenize(token_ids).context("decode output")?;
    println!("Output text:\n{out}");

    Ok(())
}
