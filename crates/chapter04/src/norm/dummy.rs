use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::Tensor;

#[derive(Clone, Copy, Debug, Module)]
pub struct DummyLayerNorm;

impl DummyLayerNorm {
    pub fn forward<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x
    }

    pub fn new(_embed_dim: usize, _eps: Option<f64>) -> Self {
        Self {}
    }
}
