use std::collections::HashSet;

use anyhow::Context;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::{DType, Tensor};
use chapter04::{GPT_124M, utils};
use tiktoken::ext::Encoding;

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    B::seed(123);

    let device = &<B as Backend>::Device::default();

    let start_context = "Hello, I am";

    let tokenizer = Encoding::gpt2();
    let encoded = {
        let allowed_specials = HashSet::default();
        tokenizer.encode(start_context, &allowed_specials)
    };

    let encoded_tensor = Tensor::<B, 1, Int>::from_ints(encoded.as_slice(), device).unsqueeze::<2>();

    let model = GPT_124M.init::<B>(device);

    let infer = model.valid();

    let out = utils::generate_text_simple(&infer, encoded_tensor.valid(), 6, GPT_124M.context_length);
    println!("Output: {out}");
    println!("Output length: {:?}", out.dims()[1]);

    let indices: Vec<u32> = out
        .squeeze::<1>(0)
        .to_data()
        .convert_dtype(DType::U32)
        .into_vec()
        .map_err(|err| anyhow::anyhow!("convert out indices: {err:?}"))?;
    let decoded_text = tokenizer.decode_str(&indices).context("decode output indices")?;
    println!("\nDecoded text: {decoded_text}");

    Ok(())
}
