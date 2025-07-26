use std::path::Path;

use anyhow::Context as _;
use burn::backend::LibTorch;
use burn::prelude::Backend;
use chapter04::GptModel;
use chapter05::gpt2;
use chapter05::utils::{self, Tokenizer as _};
use tiktoken::ext::Encoding;

type B = LibTorch;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let data_dir = &Path::new("gpt2/gpt2/124M");
    let (settings, params) = gpt2::load_settings_and_params(&data_dir).expect("load gpt2 config");

    let mut model = GptModel::<B>::new(&settings, device);

    gpt2::load_weights_into_gpt2(params, &mut model).context("load weights into model")?;

    let tokenizer = Encoding::gpt2();
    let idx = tokenizer.tokenize("Every effort moves you").to_device(device);

    B::seed(123);

    let temperature = 1.5.into();
    let token_ids = utils::generate(&model, idx, 25, settings.context_length, temperature, Some(50), None);
    let out = tokenizer.detokenize(token_ids).context("decode output")?;
    println!("Output text:\n{out}");

    Ok(())
}
