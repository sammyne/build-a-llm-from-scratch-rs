use std::fs;

use anyhow::Context;
use burn::backend::LibTorch;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use chapter04::{Config, GptModel};
use chapter05::config::GPT_124M;
use chapter05::utils::{self, Tokenizer};
use tiktoken::ext::Encoding;

// type B = NdArray<f32>;
type B = LibTorch;

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
        .no_grad();

    let idx = tokenizer.tokenize("Every effort moves you").to_device(device);

    let token_ids = utils::generate(&model, idx, 15, GPT_124M.context_length, None, 25.into(), None);

    let out = tokenizer.detokenize(token_ids).context("decode output")?;
    println!("Output text: {out}");

    Ok(())
}

fn load(c: &Config, path: &str, device: &D) -> anyhow::Result<GptModel<B>> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    GptModel::<B>::new(c, device)
        .load_file(path, &recorder, device)
        .context("load model")
}
