use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::{Tensor, activation};

#[derive(Module, Debug)]
pub struct SelfAttentionV2<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
}

impl<B: Backend> SelfAttentionV2<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // let keys = x.clone().matmul(self.k.forward(x));
        // let queries = x.clone().matmul(self.q.forward(x));
        // let values = x.clone().matmul(self.v.forward(x));

        let keys = self.k.clone().forward(x.clone());
        let queries = self.q.clone().forward(x.clone());
        let values = self.v.clone().forward(x.clone());

        let dk = *keys.dims().last().expect("get k's last dim") as f32;

        let attn_scores = queries.matmul(keys.transpose());
        let dim = attn_scores.dims().len() - 1;
        let attn_weights = activation::softmax(attn_scores / dk.sqrt(), dim);

        let context_vec = attn_weights.matmul(values);

        context_vec
    }

    pub fn new(d_in: usize, d_out: usize, qkv_bias: bool) -> Self {
        let c = LinearConfig::new(d_in, d_out).with_bias(qkv_bias);
        let device = B::Device::default();

        let q = c.init(&device);
        let k = c.init(&device);
        let v = c.init(&device);

        println!("q: {:?}\n, k: {:?}\n, v: {:?}", q, k, v);

        Self { q, k, v }
    }
}
