use std::usize;

use anyhow::Context;
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use chapter02::verdict;
use chapter05::utils::Tokenizer;
use tiktoken::ext::Encoding;

type B = Autodiff<NdArray<f32>>;
// 使用 PyTorch 后端加速明显。
// type B = Autodiff<LibTorch>;

fn main() -> anyhow::Result<()> {
    let tokenizer = Encoding::gpt2();

    let text_data = verdict::load().context("load verdict")?;

    let tokens: Tensor<B, 2, Int> = tokenizer.tokenize(&text_data);
    let total_tokens: usize = tokens.shape().dims.iter().product();
    println!("Characters: {}", text_data.len());
    println!("Tokens: {total_tokens}");

    Ok(())
}
