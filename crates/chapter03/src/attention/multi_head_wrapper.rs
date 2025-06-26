use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::attention::CausalAttention;

#[derive(Module, Debug)]
pub struct MultiHeadAttentionWrapper<B: Backend> {
    pub heads: Vec<CausalAttention<B>>,
}

impl<B: Backend> MultiHeadAttentionWrapper<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let tensors: Vec<_> = self.heads.iter().map(|h| h.forward(x.clone())).collect();
        let dim = tensors[0].dims().len() - 1;
        Tensor::cat(tensors, dim)
    }

    pub fn new(d_in: usize, d_out: usize, context_length: usize, dropout: f64, nheads: usize, qkv_bias: bool) -> Self {
        let heads: Vec<_> = (0..nheads)
            .map(|_| CausalAttention::new(d_in, d_out, context_length, dropout, qkv_bias))
            .collect();
        Self { heads }
    }
}
