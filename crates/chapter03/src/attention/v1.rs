use burn::module::{Module, Param};
use burn::prelude::Backend;
use burn::tensor::{Distribution, Tensor, activation};

/// Listing 3.1 A compact self-attention class
#[derive(Module, Debug)]
pub struct SelfAttentionV1<B: Backend> {
    wq: Param<Tensor<B, 2>>,
    wk: Param<Tensor<B, 2>>,
    wv: Param<Tensor<B, 2>>,
}

impl<B: Backend> SelfAttentionV1<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let keys = x.clone().matmul(self.wk.clone().into_value());
        let queries = x.clone().matmul(self.wq.clone().into_value());
        let values = x.clone().matmul(self.wv.clone().into_value());

        let dk = keys.dims()[1] as f32;

        let attn_scores = queries.matmul(keys.transpose());
        let dim = attn_scores.dims().len() - 1;
        let attn_weights = activation::softmax(attn_scores / dk.sqrt(), dim);

        let context_vec = attn_weights.matmul(values);

        context_vec
    }

    pub fn new(d_in: usize, d_out: usize) -> Self {
        let device = B::Device::default();
        let distribution = Distribution::Uniform(0.0, 1.0);
        let wq = Tensor::<B, 2_>::random([d_in, d_out], distribution, &device);
        let wk = Tensor::<B, 2_>::random([d_in, d_out], distribution, &device);
        let wv = Tensor::<B, 2_>::random([d_in, d_out], distribution, &device);

        Self {
            wq: Param::from_tensor(wq),
            wk: Param::from_tensor(wk),
            wv: Param::from_tensor(wv),
        }
    }
}
