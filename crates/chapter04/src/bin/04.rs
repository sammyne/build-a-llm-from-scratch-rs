use burn::backend::{Autodiff, NdArray};
use burn::prelude::Backend;
use burn::tensor::{Distribution, Tensor};
use chapter04::{FeedForward, GPT_124M};

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    let device = <B as Backend>::Device::default();

    let ffn = FeedForward::<B>::new(&GPT_124M);

    let x = Tensor::<B, 3>::random([2, 3, 768], Distribution::Uniform(0.0, 0.1), &device);

    let x = ffn.forward(x);
    println!("x.shape: {:?}", x.shape());

    Ok(())
}
