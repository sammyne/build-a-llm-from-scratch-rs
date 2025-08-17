use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::{Tensor, activation};

#[derive(Module, Debug)]
pub struct SelfAttentionV2<B: Backend> {
    pub wq: Linear<B>,
    pub wk: Linear<B>,
    pub wv: Linear<B>,
}

impl<B: Backend> SelfAttentionV2<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let keys = self.wk.clone().forward(x.clone());
        let queries = self.wq.clone().forward(x.clone());
        let values = self.wv.clone().forward(x.clone());

        let dk = keys.dims()[1] as f32;

        let attn_scores = queries.matmul(keys.transpose());

        let dim = attn_scores.dims().len() - 1;
        let attn_weights = activation::softmax(attn_scores / dk.sqrt(), dim);

        let context_vec = attn_weights.matmul(values);

        context_vec
    }

    pub fn new(d_in: usize, d_out: usize, qkv_bias: bool) -> Self {
        let c = LinearConfig::new(d_in, d_out).with_bias(qkv_bias);
        let device = B::Device::default();

        let wq = c.init(&device);
        let wk = c.init(&device);
        let wv = c.init(&device);

        Self { wq, wk, wv }
    }
}
