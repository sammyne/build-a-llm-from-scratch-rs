use std::marker::PhantomData;

use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::Tensor;

#[derive(Debug, Module)]
pub struct DummyLayerNorm<B: Backend> {
    _p: PhantomData<B>,
}

impl<B: Backend> DummyLayerNorm<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x
    }

    pub fn new(normalized_shape: usize, eps: Option<f64>) -> Self {
        Self { _p: PhantomData }
    }
}
