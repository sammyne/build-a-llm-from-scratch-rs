use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use chapter04::GPT_124M;

type B = Autodiff<NdArray<f32>>;

fn main() {
    B::seed(123);

    let device = &<B as Backend>::Device::default();

    let model = GPT_124M.init::<B>(device);

    println!("Token embedding layer shape: {:?}", model.tok_emb.weight.shape());
    println!("Output layer shape: {:?}", model.out_head.weight.shape());
}
