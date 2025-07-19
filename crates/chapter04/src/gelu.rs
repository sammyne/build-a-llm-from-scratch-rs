use std::f64::consts::PI;

use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::Tensor;

#[derive(Clone, Copy, Debug, Module)]
pub struct Gelu;

impl Gelu {
    pub fn forward<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // gelu(x)=0.5*x*(1+Tanh(0.5*√2/π*(x+0.044715*x^3)))
        let tanh = (x.clone() + x.clone().powi_scalar(3).mul_scalar(0.044715))
            .mul_scalar((2.0 / PI).sqrt())
            .tanh();

        x.mul_scalar(0.5) * (tanh.add_scalar(1.0))
    }
}
