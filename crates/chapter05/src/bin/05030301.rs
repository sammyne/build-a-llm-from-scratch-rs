use std::fs;

use anyhow::Context;
use burn::backend::LibTorch;
use burn::prelude::*;
use chapter05::config::GPT_124M;
use chapter05::utils::{self, GenerateOptions, Tokenizer};
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

    let model = GPT_124M
        .load::<B>("gpt_124m_trained.mpk", device)
        .context("load model")?
        .no_grad();

    let idx = tokenizer.tokenize("Every effort moves you").to_device(device);

    let opts = GenerateOptions::new(15, GPT_124M.context_length)
        .with_topk(25.into())
        .with_temperature(1.4);
    let token_ids = utils::generate(&model, idx, opts);

    let out = tokenizer.detokenize(token_ids).context("decode output")?;
    println!("Output text: {out}");

    Ok(())
}
