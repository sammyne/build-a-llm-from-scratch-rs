use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::{Config, Gelu};

#[derive(Debug, Module)]
pub struct FeedForward<B: Backend> {
    linear1: Linear<B>,
    gelu: Gelu,
    linear2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.linear2.forward(x);

        x
    }

    pub fn new(c: &Config) -> Self {
        let device = &B::Device::default();

        let linear1 = LinearConfig::new(c.emb_dim, 4 * c.emb_dim).init(device);
        let gelu = Gelu;
        let linear2 = LinearConfig::new(4 * c.emb_dim, c.emb_dim).init(device);

        Self { linear1, gelu, linear2 }
    }
}
