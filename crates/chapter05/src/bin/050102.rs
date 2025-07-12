use anyhow::Context;
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::activation;
use chapter04::GptModel;
use chapter05::config::GPT_124M;
use chapter05::utils::{self, Tokenizer};
use tiktoken::ext::Encoding;

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    B::seed(123);
    let device = &<B as Backend>::Device::Cpu;

    demo_manual_cross_entropy();

    let model = GptModel::<B>::new(GPT_124M, device).no_grad();

    let device = <B as Backend>::Device::default();

    // ["every effort moves", "I really like"]
    let inputs = Tensor::<B, 2, Int>::from_ints([[16833, 3626, 6100], [40, 1107, 588]], &device);
    // [" effort moves you", " really like chocolate"]
    let targets = Tensor::<B, 2, Int>::from_ints([[3626, 6100, 345], [1107, 588, 11311]], &device);

    let logits = model.forward(inputs);
    let dim = logits.dims().len() - 1;
    let probas = activation::softmax(logits.clone(), dim);
    println!("{:?}", probas.shape());

    let token_ids = probas.clone().argmax(dim);
    println!("Token IDs: {}", token_ids);

    let tokenizer = Encoding::gpt2();

    let t0 = tokenizer
        .detokenize(targets.clone().slice(s![0]))
        .context("de-tokenize target[0]")?;
    let o0 = tokenizer
        .detokenize(token_ids.clone().slice(s![0]).flatten::<2>(1, 2))
        .context("de-tokenize target[0]")?;
    println!("Targets batch 1: {t0}");
    println!("Outputs batch 1: {o0}");

    println!("probas: {probas}");

    let mut ps = vec![];
    for text_idx in [0, 1] {
        let target_idx = targets.clone().slice(s![text_idx]).transpose();
        let target_probas = probas
            .clone()
            .slice(s![text_idx, 0..=2])
            .squeeze::<2>(0)
            .gather(1, target_idx)
            .flatten::<1>(0, 1);
        println!("Text {text_idx}: {target_probas}");
        ps.push(target_probas);
    }

    let log_probas = Tensor::cat(ps, 0).log();
    println!("log-probas: {log_probas}");

    let loss = utils::cross_entropy(logits, targets);
    println!("loss: {loss}");

    Ok(())
}

fn demo_manual_cross_entropy() {
    let device = <B as Backend>::Device::default();

    let p1 = Tensor::<B, 1>::from_floats([7.4541e-05, 3.1061e-05, 1.1563e-05], &device);
    let p2 = Tensor::<B, 1>::from_floats([1.0337e-05, 5.6776e-05, 4.7559e-06], &device);

    let log_probas = Tensor::cat(vec![p1, p2], 0).log();
    println!("log-probas: {log_probas}");

    let avg_log_probas = log_probas.mean().into_scalar();
    println!("mean: {avg_log_probas}");
    println!("-mean: {}", -avg_log_probas);
}
