use burn::backend::{Autodiff, NdArray};
use burn::prelude::Backend;
use burn::tensor::{Distribution, Tensor};
use chapter04::{GPT_124M, TransformerBlockConfig};

type B = Autodiff<NdArray<f32>>;

fn main() {
    B::seed(123);

    let device = &<B as Backend>::Device::default();

    let x = Tensor::<B, 3>::random([2, 4, 768], Distribution::Uniform(0.0, 1.0), device);

    let block = TransformerBlockConfig::new(
        GPT_124M.context_length,
        GPT_124M.emb_dim,
        GPT_124M.nheads,
        GPT_124M.drop_rate,
        GPT_124M.qkv_bias,
    )
    .init::<B>(device);

    let output = block.forward(x.clone());

    println!("Input shape: {:?}", x.shape());
    println!("Output shape: {:?}", output.shape());
}
