use anyhow::Context as _;
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::activation;
use chapter05::config::GPT_124M;
use chapter05::utils::Tokenizer as _;
use tiktoken::ext::Encoding;

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    B::seed(123);
    let device = &<B as Backend>::Device::Cpu;

    // ["every effort moves", "I really like"]
    let inputs = Tensor::<B, 2, Int>::from_ints([[16833, 3626, 6100], [40, 1107, 588]], &device);
    // [" effort moves you", " really like chocolate"]
    let targets = Tensor::<B, 2, Int>::from_ints([[3626, 6100, 345], [1107, 588, 11311]], &device);

    let model = GPT_124M.init::<B>(device).no_grad();

    let logits = model.forward(inputs);
    let dim = logits.dims().len() - 1;
    let probas = activation::softmax(logits.clone(), dim);

    let token_ids = probas.clone().argmax(dim);

    let tokenizer = Encoding::gpt2();

    let t0 = tokenizer
        .detokenize(targets.clone().slice(s![0]))
        .context("de-tokenize target[0]")?;
    let o0 = tokenizer
        .detokenize(token_ids.clone().slice(s![0]).flatten::<2>(1, 2))
        .context("de-tokenize target[0]")?;
    println!("Targets batch 1: {t0}");
    println!("Outputs batch 1: {o0}");

    Ok(())
}
