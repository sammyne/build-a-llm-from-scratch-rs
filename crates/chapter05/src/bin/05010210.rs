use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use chapter05::config::GPT_124M;
use chapter05::utils;

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

    let logits_flat = logits.flatten::<2>(0, 1);
    let targets_flat = targets.flatten::<1>(0, 1);

    let loss = utils::cross_entropy(logits_flat, targets_flat);
    println!("{loss}");
}
