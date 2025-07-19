use std::fs;

use anyhow::Context;
use burn::backend::{Autodiff, LibTorch};
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use chapter04::{Config, GptModel, utils};
use chapter05::config::GPT_124M;
use chapter05::utils::Tokenizer;
use tiktoken::ext::Encoding;

// type B = Autodiff<NdArray<f32>>;
type B = Autodiff<LibTorch>;

type D = <B as Backend>::Device;

fn main() -> anyhow::Result<()> {
    if !fs::exists("gpt_124m_trained.mpk").context("check model path")? {
        anyhow::bail!("train GPT-124M model first by running `cargo run --bin 0502`");
    }

    let tokenizer = Encoding::gpt2();

    B::seed(123);
    let device = &D::Cpu;

    let model = load(GPT_124M, "gpt_124m_trained.mpk", device)
        .context("load model")?
        .valid();

    let idx = tokenizer.tokenize("Every effort moves you").to_device(device);

    let token_ids = utils::generate_text_simple(&model, idx, 25, GPT_124M.context_length);
    let out = tokenizer.detokenize(token_ids).context("decode output")?;
    println!("Output text:\n{out}");

    Ok(())
}

fn load(c: &Config, path: &str, device: &D) -> anyhow::Result<GptModel<B>> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    GptModel::<B>::new(c, device)
        .load_file(path, &recorder, device)
        .context("load model")
}
