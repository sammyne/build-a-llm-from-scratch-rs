use burn::backend::{Autodiff, NdArray};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::Backend;
use burn::tensor::{Distribution, Tensor};

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    let device = <B as Backend>::Device::default();

    B::seed(123);

    let batch_example = Tensor::<B, 2>::random([2, 5], Distribution::Uniform(0.0, 1.0), &device);

    let layer = Sequential {
        linear: LinearConfig::new(5, 6).init(&device),
        relu: Relu::new(),
    };

    let out = layer.forward(batch_example.clone());
    println!("{out}");

    Ok(())
}

pub struct Sequential<B: Backend> {
    linear: Linear<B>,
    relu: Relu,
}

impl<B: Backend> Sequential<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear.forward(x);
        let x = self.relu.forward(x);
        x
    }
}
