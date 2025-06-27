use burn::module::{Module, Param};
use burn::nn::{Dropout, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::{Tensor, activation};

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    pub d_out: usize,
    pub nheads: usize,
    pub head_dim: usize,

    pub q: Linear<B>,
    pub k: Linear<B>,
    pub v: Linear<B>,
    pub out_proj: Linear<B>,
    pub dropout: Dropout,
    pub mask: Param<Tensor<B, 4>>,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (b, ntokens, d_in) = {
            let s = x.shape().dims;
            (s[0], s[1], s[2])
        };

        let keys = self.k.forward(x.clone());
        let queries = self.q.forward(x.clone());
        let values = self.v.forward(x.clone());

        let keys = keys.reshape::<4, _>([b, ntokens, self.nheads, self.head_dim]);
        let values = values.reshape::<4, _>([b, ntokens, self.nheads, self.head_dim]);
        let queries = queries.reshape::<4, _>([b, ntokens, self.nheads, self.head_dim]);

        let keys = keys.swap_dims(1, 2);
        let queries = queries.swap_dims(1, 2);
        let values = values.swap_dims(1, 2);

        let dk = *keys.dims().last().expect("get k's last dim") as f32;

        let attn_scores = queries.matmul(keys.transpose());
        let mask = self.mask.val().bool().slice([..1, ..ntokens, ..ntokens]);

        let attn_scores = attn_scores.mask_fill(mask, f32::NEG_INFINITY);

        let dim = attn_scores.dims().len() - 1;
        let attn_weights = activation::softmax(attn_scores / dk.sqrt(), dim);
        let attn_weights = self.dropout.forward(attn_weights);

        let context_vec = attn_weights.matmul(values).swap_dims(1, 2);

        let context_vec = context_vec.reshape::<3, _>([b, ntokens, self.d_out]);

        let context_vec = self.out_proj.forward(context_vec);

        context_vec
    }

    pub fn new(d_in: usize, d_out: usize, context_length: usize, dropout: f64, nheads: usize, qkv_bias: bool) -> Self {
        assert_eq!(0, d_out % nheads, "d_out must be divisible by num_heads");

        let head_dim = d_out / nheads;

        let c = LinearConfig::new(d_in, d_out).with_bias(qkv_bias);
        let device = B::Device::default();

        let q = c.init(&device);
        let k = c.init(&device);
        let v = c.init(&device);

        let out_proj = LinearConfig::new(d_out, d_out).init(&device);

        let dropout = Dropout { prob: dropout };

        let mask = Tensor::<B, 2>::ones([context_length, context_length], &device)
            .triu(1)
            .unsqueeze::<4>();
        let mask = Param::from_tensor(mask);

        Self {
            d_out,
            nheads,
            head_dim,

            q,
            k,
            v,
            out_proj,
            dropout,
            mask,
        }
    }
}
