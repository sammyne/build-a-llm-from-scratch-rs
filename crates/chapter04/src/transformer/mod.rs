mod dummy;

use burn::module::Module;
use burn::nn::Dropout;
use burn::prelude::*;
use burn::tensor::Tensor;
use chapter03::attention::MultiHeadAttention;
pub use dummy::*;

use crate::{Config, FeedForward, LayerNorm};

#[derive(Debug, Module)]
pub struct TransformerBlock<B: Backend> {
    attn: MultiHeadAttention<B>,
    ff: FeedForward<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    drop_shortcut: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let shortcut = x.clone();

        let x = self.norm1.forward(x);
        let x = self.attn.forward(x);
        let x = self.drop_shortcut.forward(x);
        let x = x + shortcut;

        let shortcut = x.clone();
        let x = self.norm2.forward(x);
        let x = self.ff.forward(x);
        let x = self.drop_shortcut.forward(x);
        let x = x + shortcut;

        x
    }

    pub fn new(c: &Config) -> Self {
        let attn = MultiHeadAttention::new(
            c.emb_dim,
            c.emb_dim,
            c.context_length,
            c.drop_rate,
            c.nheads,
            c.qkv_bias,
        );
        let ff = FeedForward::new(c);
        let norm1 = LayerNorm::<_, 3>::new(c.emb_dim);
        let norm2 = LayerNorm::<_, 3>::new(c.emb_dim);
        let drop_shortcut = Dropout { prob: c.drop_rate };

        Self {
            attn,
            ff,
            norm1,
            norm2,
            drop_shortcut,
        }
    }
}
