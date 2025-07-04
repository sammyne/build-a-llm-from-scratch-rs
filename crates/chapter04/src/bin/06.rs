use burn::backend::{Autodiff, NdArray};
use burn::prelude::Backend;
use burn::tensor::{Distribution, Tensor};
use chapter04::{TransformerBlock, GPT_124M};

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
  B::seed(123);

  let device = <B as Backend>::Device::default();

  let x = Tensor::<B,3>::random([2,4,768], Distribution::Uniform(0.0, 1.0), &device);

  let block = TransformerBlock::<B>::new(&GPT_124M);

  let output= block.forward(x.clone());

  println!("Input shape: {:?}", x.shape());
  println!("Output shape: {:?}", output.shape());



  Ok(())
}
