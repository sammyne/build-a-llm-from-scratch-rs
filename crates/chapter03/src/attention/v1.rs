use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::{Distribution, Tensor, activation};

#[derive(Module, Debug)]
pub struct SelfAttentionV1<B: Backend> {
    q: Tensor<B, 2>,
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
}

impl<B: Backend> SelfAttentionV1<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let keys = x.clone().matmul(self.k.clone());
        let queries = x.clone().matmul(self.q.clone());
        let values = x.clone().matmul(self.v.clone());

        let dk = *keys.dims().last().expect("get k's last dim") as f32;

        let attn_scores = queries.matmul(keys.transpose());
        let dim = attn_scores.dims().len() - 1;
        let attn_weights = activation::softmax(attn_scores / dk.sqrt(), dim);

        let context_vec = attn_weights.matmul(values);

        context_vec
    }

    pub fn new(d_in: usize, d_out: usize) -> Self {
        let device = B::Device::default();
        let distribution = Distribution::Uniform(0.0, 1.0);
        let q = Tensor::<B, 2_>::random([d_in, d_out], distribution, &device);
        let k = Tensor::<B, 2_>::random([d_in, d_out], distribution, &device);
        let v = Tensor::<B, 2_>::random([d_in, d_out], distribution, &device);

        Self { q, k, v }
    }
}
