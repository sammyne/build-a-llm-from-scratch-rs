use burn::module::Module;
use burn::nn::{Dropout, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Tensor, activation};

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    pub d_out: usize,
    pub nheads: usize,
    pub head_dim: usize,

    pub wq: Linear<B>,
    pub wk: Linear<B>,
    pub wv: Linear<B>,
    pub out_proj: Linear<B>,
    pub dropout: Dropout,
    pub mask: Tensor<B, 4>,
}

/// 仿照 https://docs.rs/burn-core/0.18.0/src/burn_core/nn/linear.rs.html#14
/// 目的是缩短构造 MultiHeadAttention 的形参列表长度，并支持字段的默认值。
#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    pub d_in: usize,
    pub d_out: usize,
    pub context_length: usize,
    pub dropout: f64,
    pub nheads: usize,
    #[config(default = false)]
    pub qkv_bias: bool,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// 输入的维度为 (batch-size, num-tokens, embedding-dim)。
    /// 输出的维度为 (batch-size, num-tokens, d-out)。
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (b, ntokens) = {
            let s = x.shape().dims;
            (s[0], s[1])
        };

        // 维度变化：(batch-size, num-tokens, embedding-dim) -> (batch-size, num-tokens, d-out)
        let keys = self.wk.forward(x.clone());
        let queries = self.wq.forward(x.clone());
        let values = self.wv.forward(x.clone());

        // 维度变化：(batch-size, num-tokens, d-out) -> (batch-size, num-tokens, num-heads, head-dim)
        let keys = keys.reshape::<4, _>([b, ntokens, self.nheads, self.head_dim]);
        let values = values.reshape::<4, _>([b, ntokens, self.nheads, self.head_dim]);
        let queries = queries.reshape::<4, _>([b, ntokens, self.nheads, self.head_dim]);

        // 维度变化：(batch-size, num-tokens, num-heads, head-dim) -> (batch-size, num-heads, num-tokens, head-dim)
        let keys = keys.swap_dims(1, 2);
        let queries = queries.swap_dims(1, 2);
        let values = values.swap_dims(1, 2);

        let dk = *keys.dims().last().expect("get k's last dim") as f32;

        let attn_scores = queries.matmul(keys.transpose());
        let mask = self.mask.clone().bool().slice(s![.., .., ..ntokens, ..ntokens]);

        let attn_scores = attn_scores.mask_fill(mask, f32::NEG_INFINITY);

        let dim = attn_scores.dims().len() - 1;
        let attn_weights = activation::softmax(attn_scores / dk.sqrt(), dim);
        let attn_weights = self.dropout.forward(attn_weights);

        // 维度变化：(batch-size, num-heads, num-tokens, head-dim) -> (batch-size, num-tokens, num-heads, head-dim)
        let context_vec = attn_weights.matmul(values).swap_dims(1, 2);

        // 维度变化：(batch-size, num-tokens, num-heads, head-dim) -> (batch-size, num-tokens, d-out)
        let context_vec = context_vec.reshape::<3, _>([b, ntokens, self.d_out]);

        let context_vec = self.out_proj.forward(context_vec);

        context_vec
    }
}

impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        let Self {
            d_in,
            d_out,
            context_length,
            dropout,
            nheads,
            qkv_bias,
        } = *self;

        assert_eq!(0, d_out % nheads, "d_out must be divisible by num_heads");

        let head_dim = d_out / nheads;

        let c = LinearConfig::new(d_in, d_out).with_bias(qkv_bias);

        let wq = c.init(device);
        let wk = c.init(device);
        let wv = c.init(device);

        let out_proj = LinearConfig::new(d_out, d_out).init(device);

        let dropout = Dropout { prob: dropout };

        let mask = Tensor::<B, 2>::ones([context_length, context_length], device)
            .triu(1)
            .unsqueeze::<4>();

        MultiHeadAttention {
            d_out,
            nheads,
            head_dim,

            wq,
            wk,
            wv,
            out_proj,
            dropout,
            mask,
        }
    }
}
