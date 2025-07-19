use std::fs;

use anyhow::Context;
use burn::backend::LibTorch;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::activation;
use chapter04::{Config, GptModel};
use chapter05::config::GPT_124M;
use chapter05::utils::Tokenizer;
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

    let token_ids = generate(&model, idx, 15, GPT_124M.context_length, None, 25.into(), None);

    let out = tokenizer.detokenize(token_ids).context("decode output")?;
    println!("Output text: {out}");

    Ok(())
}

fn generate(
    model: &GptModel<B>,
    mut idx: Tensor<B, 2, Int>,
    max_new_tokens: usize,
    context_size: usize,
    temperature: Option<f32>,
    topk: Option<usize>,
    eos_id: Option<usize>,
) -> Tensor<B, 2, Int> {
    let context_size = context_size as i32;

    for _ in 0..max_new_tokens {
        let idx_cond = idx.clone().slice(s![.., -context_size..]);

        let logits = model.forward(idx_cond);
        let mut logits = logits.slice(s![.., -1, ..]).squeeze(1);

        if let Some(k) = topk {
            let top_logits = logits.clone().squeeze::<1>(0).topk(k, 0);
            let v = top_logits.min().into_scalar();
            let discarded = logits.clone().lower(logits.clone().full_like(v));
            logits = logits.mask_fill(discarded, f32::NEG_INFINITY);
        }

        let dim = logits.dims().len() - 1;
        let idx_next = match temperature {
            Some(t) => {
                logits = logits / t;
                let probas = activation::softmax(logits.clone(), dim);
                todo!("补充 multinomial 的实现");
            }
            None => logits.argmax(dim),
        };

        match eos_id {
            Some(v) if idx_next.clone().squeeze::<1>(0).into_scalar() == v as i64 => break,
            _ => {}
        }

        println!("hello");
        idx = Tensor::cat(vec![idx, idx_next], 1);
    }

    idx
}

fn load(c: &Config, path: &str, device: &D) -> anyhow::Result<GptModel<B>> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    GptModel::<B>::new(c, device)
        .load_file(path, &recorder, device)
        .context("load model")
}
