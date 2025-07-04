mod dummy;

use burn::module::{Module, Param};
use burn::prelude::Backend;
use burn::tensor::Tensor;
pub use dummy::*;

#[derive(Debug, Module)]
pub struct LayerNorm<B: Backend, const D: usize = 3> {
    eps: f64,
    scale: Param<Tensor<B, D>>,
    shift: Param<Tensor<B, D>>,
}

impl<B: Backend, const D: usize> LayerNorm<B, D> {
    pub fn forward(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dim = x.dims().len() - 1;

        let (var, mean) = x.clone().var_mean_bias(dim);
        let norm_x = (x - mean) / (var + self.eps).sqrt();

        self.scale.val() * norm_x + self.shift.val()
    }

    pub fn new(embed_dim: usize) -> Self {
        let eps = 1e-5;

        let device = B::Device::default();

        let scale = Param::from_tensor(Tensor::<B, 1>::ones([embed_dim], &device).unsqueeze::<D>());
        let shift = Param::from_tensor(Tensor::<B, 1>::zeros([embed_dim], &device).unsqueeze::<D>());

        Self { eps, scale, shift }
    }
}
