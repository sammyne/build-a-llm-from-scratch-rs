use burn::backend::{Autodiff, NdArray};
use burn::nn::{Linear, Relu};
use burn::prelude::Backend;
use burn::tensor::{Distribution, Tensor};
use chapter04::LayerNormConfig;

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    let device = &<B as Backend>::Device::default();

    B::seed(123);

    let batch_example = Tensor::<B, 2>::random([2, 5], Distribution::Uniform(0.0, 1.0), &device);

    let ln = LayerNormConfig::new(5).init(device);
    let out_ln = ln.forward(batch_example.clone());
    let dim = out_ln.dims().len() - 1;
    let (var, mean) = out_ln.var_mean_bias(dim);
    println!("mean: {mean}");
    println!("var: {var}");

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
