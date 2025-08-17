mod dummy;

use burn::config::Config;
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

#[derive(Config, Copy, Debug)]
pub struct LayerNormConfig {
    pub embed_dim: usize,
    #[config(default = 1e-5)]
    pub eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dim = x.dims().len() - 1;

        let (var, mean) = x.clone().var_mean_bias(dim);
        let norm_x = (x - mean) / (var + self.eps).sqrt();

        self.scale.val().unsqueeze() * norm_x + self.shift.val().unsqueeze()
    }
}

impl LayerNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        let scale = Param::from_tensor(Tensor::<B, 1>::ones([self.embed_dim], device));
        let shift = Param::from_tensor(Tensor::<B, 1>::zeros([self.embed_dim], device));

        LayerNorm {
            eps: self.eps,
            scale,
            shift,
        }
    }
}
