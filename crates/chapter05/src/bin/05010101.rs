use anyhow::Context;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::prelude::*;
use chapter05::config::GPT_124M;
use chapter05::utils::Tokenizer;
use tiktoken::ext::Encoding;

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    B::seed(123);
    let device = &<B as Backend>::Device::Cpu;

    let model = GPT_124M.init::<B>(device);
    let model = model.valid();

    let start_context = "Hello, I am";

    let tokenizer = Encoding::gpt2();

    let idx = tokenizer.tokenize(start_context);

    let token_ids = chapter04::utils::generate_text_simple(&model, idx, 10, GPT_124M.context_length);

    let decoded = tokenizer.detokenize(token_ids).context("de-tokenize output")?;
    println!("Output text: {decoded}");

    Ok(())
}
