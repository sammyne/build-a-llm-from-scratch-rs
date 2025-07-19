mod dummy;

use burn::module::{Module, Param};
use burn::prelude::Backend;
use burn::tensor::Tensor;
pub use dummy::*;

#[derive(Debug, Module)]
pub struct LayerNorm<B: Backend> {
    pub eps: f64,
    pub scale: Param<Tensor<B, 1>>,
    pub shift: Param<Tensor<B, 1>>,
}

impl<B: Backend> LayerNorm<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dim = x.dims().len() - 1;

        let (var, mean) = x.clone().var_mean_bias(dim);
        let norm_x = (x - mean) / (var + self.eps).sqrt();

        self.scale.val().unsqueeze() * norm_x + self.shift.val().unsqueeze()
    }

    pub fn new(embed_dim: usize) -> Self {
        let eps = 1e-5;

        let device = B::Device::default();

        let scale = Param::from_tensor(Tensor::<B, 1>::ones([embed_dim], &device));
        let shift = Param::from_tensor(Tensor::<B, 1>::zeros([embed_dim], &device));

        Self { eps, scale, shift }
    }
}
