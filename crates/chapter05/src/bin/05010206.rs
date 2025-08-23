use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::activation;
use chapter05::config::GPT_124M;

type B = Autodiff<NdArray<f32>>;

fn main() {
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

    let mut ps = vec![];
    for text_idx in [0, 1] {
        let target_idx = targets.clone().slice(s![text_idx]).transpose();
        let target_probas = probas
            .clone()
            .slice(s![text_idx, 0..=2])
            .squeeze::<2>(0)
            .gather(1, target_idx)
            .flatten::<1>(0, 1);
        ps.push(target_probas);
    }

    let log_probas = Tensor::cat(ps, 0).log();

    let avg_log_probas = log_probas.mean().into_scalar();
    println!("{avg_log_probas}");
}
