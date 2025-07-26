use std::collections::HashSet;

use anyhow::Context;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::tensor::{DType, Tensor};
use chapter04::{GPT_124M, GptModel, utils};
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
    println!("encoded: {encoded:?}");

    let encoded_tensor = Tensor::<B, 1, Int>::from_ints(encoded.as_slice(), &device).unsqueeze::<2>();
    println!("encoded-tensor.shape: {:?}", encoded_tensor.shape());

    let model = GptModel::<B>::new(&GPT_124M, device);

    let infer = model.valid();

    let out = utils::generate_text_simple(&infer, encoded_tensor.inner(), 6, GPT_124M.context_length);
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
