mod dummy;

use burn::module::{Module, Param};
use burn::prelude::Backend;
use burn::tensor::Tensor;
pub use dummy::*;

#[derive(Debug, Module)]
pub struct LayerNorm<B: Backend> {
    eps: f64,
    scale: Param<Tensor<B, 2>>,
    shift: Param<Tensor<B, 2>>,
}

impl<B: Backend> LayerNorm<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let dim = x.dims().len() - 1;

        let (var, mean) = x.clone().var_mean_bias(dim);
        let norm_x = (x - mean) / (var + self.eps).sqrt();

        self.scale.val() * norm_x + self.shift.val()
    }

    pub fn new(embed_dim: usize) -> Self {
        let eps = 1e-5;

        let device = B::Device::default();

        let scale = Param::from_tensor(Tensor::<B, 2>::ones([1, embed_dim], &device));
        let shift = Param::from_tensor(Tensor::<B, 2>::zeros([1, embed_dim], &device));

        Self { eps, scale, shift }
    }
}
