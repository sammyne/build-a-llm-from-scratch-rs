use burn::backend::{Autodiff, NdArray};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::Backend;
use burn::tensor::{Distribution, Tensor};
use chapter04::LayerNorm;

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    let device = <B as burn::prelude::Backend>::Device::default();

    B::seed(123);

    let batch_example = Tensor::<B, 2>::random([2, 5], Distribution::Uniform(0.0, 1.0), &device);

    let layer = Sequential {
        linear: LinearConfig::new(5, 6).init(&device),
        relu: Relu::new(),
    };

    let out = layer.forward(batch_example.clone());
    println!("{out:?}");

    let dim = out.dims().len() - 1;

    let (var, mean) = out.clone().var_mean(dim);
    println!("mean: {mean}");
    println!("var: {var}");

    let out_norm = (out - mean) / var.sqrt();
    let (var, mean) = out_norm.var_mean(dim);
    println!("mean: {mean}");
    println!("var: {var}");

    let ln = LayerNorm::<B>::new(5);
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
