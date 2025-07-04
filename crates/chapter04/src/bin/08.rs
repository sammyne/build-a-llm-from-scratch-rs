use std::collections::HashSet;

use anyhow::Context;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::{DType, Tensor, activation};
use chapter04::{GPT_124M, GptModel};
use tiktoken::ext::Encoding;

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    B::seed(123);

    let device = <B as Backend>::Device::default();

    let start_context = "Hello, I am";

    let tokenizer = Encoding::gpt2();
    let encoded = {
        let allowed_specials = HashSet::default();
        tokenizer.encode(start_context, &allowed_specials)
    };
    println!("encoded: {encoded:?}");

    let encoded_tensor = Tensor::<B, 1, Int>::from_ints(encoded.as_slice(), &device).unsqueeze::<2>();
    println!("encoded-tensor.shape: {:?}", encoded_tensor.shape());

    let model = GptModel::<B>::new(&GPT_124M);

    let infer = model.valid();

    let out = generate_text_simple(&infer, encoded_tensor.inner(), 6, GPT_124M.context_length);
    println!("Output: {out}");
    println!("Output length: {:?}", out.dims()[1]);

    let indices: Vec<u32> = out
        .squeeze::<1>(0)
        .to_data()
        .convert_dtype(DType::U32)
        .into_vec()
        .map_err(|err| anyhow::anyhow!("conv out indices: {err:?}"))?;
    let decoded_text = tokenizer.decode_str(&indices).context("decode output indices")?;
    println!("Decoded text: {decoded_text}");

    Ok(())
}

fn generate_text_simple(
    model: &GptModel<NdArray>,
    mut idx: Tensor<NdArray, 2, Int>,
    max_new_tokens: usize,
    context_size: usize,
) -> Tensor<NdArray, 2, Int> {
    let context_size = context_size as i32;
    for _ in 0..max_new_tokens {
        let idx_cond = idx.clone().slice(s![.., -context_size..]);
        let logits = model.forward(idx_cond);

        let logits = logits.slice(s![.., -1, ..]).squeeze(1);

        let dim = logits.dims().len() - 1;

        let probas = activation::softmax(logits, dim);
        let idx_next = probas.argmax(dim);
        idx = Tensor::cat(vec![idx, idx_next], 1);
    }

    idx
}
