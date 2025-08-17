use std::marker::PhantomData;

use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::Tensor;

#[derive(Debug, Default, Module)]
pub struct DummyTransformerBlock<B: Backend> {
    _p: PhantomData<B>,
}

impl<B: Backend> DummyTransformerBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x
    }

    pub fn new() -> Self {
        Self { _p: PhantomData }
    }
}
