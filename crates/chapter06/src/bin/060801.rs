use std::path::Path;

use anyhow::Context as _;
use burn::backend::{Autodiff, LibTorch};
use burn::module::{AutodiffModule, Module};
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, s};
use chapter04::{GPT_124M, GptModel};
use chapter06::dataset::{LoadCsvOptions, SpamDataset};
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;

type Device = <LibTorch as Backend>::Device;

/// 依赖 060701 训练出的模型。
fn main() -> anyhow::Result<()> {
    let model_path = Path::new("spam-classifier-model.mpk");
    if !model_path.exists() {
        anyhow::bail!("model not exists");
    }

    let device = &Device::Cpu;

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let model = GPT_124M
        .with_qkv_bias(true)
        .init::<B>(device)
        .load_file("spam-classifier-model.mpk", &recorder, device)
        .context("load model")?;
    let model = model.no_grad();

    let tokenizer = Encoding::gpt2();

    let opts = LoadCsvOptions::new("train.csv", &tokenizer, device);
    let train_dataset = SpamDataset::<B>::load_csv(opts).context("load train dataset")?;
    let max_length = train_dataset.max_length;

    const TEXT_1: &str = std::concat!(
        "You are a winner you have been specially",
        " selected to receive $1000 cash or a $2000 award."
    );
    let out = classify_review(TEXT_1, model.clone(), &tokenizer, device, max_length.into(), None);
    println!("{out}");

    Ok(())
}

fn classify_review<B: AutodiffBackend>(
    text: &str,
    model: GptModel<B>,
    tokenizer: &Encoding,
    device: &B::Device,
    max_length: Option<usize>,
    pad_token_id: Option<u32>,
) -> &'static str {
    let model = model.valid();

    // Prepare inputs to the model
    let mut input_ids = {
        let allowed = Default::default();
        tokenizer.encode(text, &allowed)
    };
    let supported_context_length = model.pos_emb.weight.dims()[0];

    let max_len = match max_length {
        None => supported_context_length,
        Some(v) => v.min(supported_context_length),
    };

    // Truncate sequences if they too long, or pad sequences to the longest sequence
    let pad_token_id = pad_token_id.unwrap_or(50256);
    input_ids.resize(max_len, pad_token_id);

    let input_tensor = Tensor::<B::InnerBackend, 1, Int>::from_ints(input_ids.as_slice(), device).unsqueeze_dim(0);

    let logits = model.no_grad().forward(input_tensor).slice(s![.., -1, ..]);
    let dim = logits.dims().len() - 1;
    let predicted_label = logits
        .argmax(dim)
        .into_data()
        .to_vec::<i64>()
        .expect("tensor as Vec<i64>")[0] as u32;

    if predicted_label == 1 { "spam" } else { "not spam" }
}
