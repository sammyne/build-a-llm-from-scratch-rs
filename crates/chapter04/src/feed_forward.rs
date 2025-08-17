use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::Gelu;

#[derive(Debug, Module)]
pub struct FeedForward<B: Backend> {
    pub linear1: Linear<B>,
    pub gelu: Gelu,
    pub linear2: Linear<B>,
}

#[derive(Config, Copy, Debug)]
pub struct FeedForwardConfig {
    /// 输入的维度。
    pub d_model: usize,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = self.gelu.forward(x);
        let x = self.linear2.forward(x);

        x
    }
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        let linear1 = LinearConfig::new(self.d_model, 4 * self.d_model).init(device);
        let gelu = Gelu;
        let linear2 = LinearConfig::new(4 * self.d_model, self.d_model).init(device);

        FeedForward { linear1, gelu, linear2 }
    }
}
