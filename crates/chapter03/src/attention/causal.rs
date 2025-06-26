use core::f32;

use burn::module::{Module, Param};
use burn::nn::{Dropout, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::{Tensor, activation};

#[derive(Module, Debug)]
pub struct CausalAttention<B: Backend> {
    pub q: Linear<B>,
    pub k: Linear<B>,
    pub v: Linear<B>,
    pub dropout: Dropout,
    pub mask: Param<Tensor<B, 3>>,
}

impl<B: Backend> CausalAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let num_tokens = x.shape().dims[1];

        //let x = x.set_require_grad(true);

        let keys = self.k.clone().forward(x.clone());
        let queries = self.q.clone().forward(x.clone());
        let values = self.v.clone().forward(x.clone());

        let dk = *keys.dims().last().expect("get k's last dim") as f32;

        let attn_scores = queries.matmul(keys.transpose());

        let mask = self.mask.val().bool().slice([..1, ..num_tokens, ..num_tokens]);
        let attn_scores = attn_scores.mask_fill(mask, f32::NEG_INFINITY);

        let dim = attn_scores.dims().len() - 1;
        let attn_weights = activation::softmax(attn_scores / dk.sqrt(), dim);
        let attn_weights = self.dropout.forward(attn_weights);

        let context_vec = attn_weights.matmul(values);

        context_vec
    }

    pub fn new(d_in: usize, d_out: usize, context_length: usize, dropout: f64, qkv_bias: bool) -> Self {
        let c = LinearConfig::new(d_in, d_out).with_bias(qkv_bias);
        let device = B::Device::default();

        let q = c.init(&device);
        let k = c.init(&device);
        let v = c.init(&device);

        let dropout = Dropout { prob: dropout };

        let mask = Tensor::<B, 2>::ones([context_length, context_length], &device)
            .triu(1)
            .unsqueeze::<3>();
        let mask = Param::from_tensor(mask);

        Self { q, k, v, dropout, mask }
    }
}
