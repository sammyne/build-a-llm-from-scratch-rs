use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::activation;
use chapter05::config::GPT_124M;

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    B::seed(123);
    let device = &<B as Backend>::Device::Cpu;

    // ["every effort moves", "I really like"]
    let inputs = Tensor::<B, 2, Int>::from_ints([[16833, 3626, 6100], [40, 1107, 588]], &device);

    let model = GPT_124M.init::<B>(device).no_grad();

    let logits = model.forward(inputs);
    let dim = logits.dims().len() - 1;
    let probas = activation::softmax(logits.clone(), dim);
    println!("{:?}", probas.shape());

    Ok(())
}
