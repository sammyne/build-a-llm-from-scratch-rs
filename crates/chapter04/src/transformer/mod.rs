mod dummy;

use burn::module::Module;
use burn::nn::Dropout;
use burn::prelude::*;
use burn::tensor::Tensor;
use chapter03::attention::{MultiHeadAttention, MultiHeadAttentionConfig};
pub use dummy::*;

use crate::{FeedForward, FeedForwardConfig, LayerNorm, LayerNormConfig};

#[derive(Debug, Module)]
pub struct TransformerBlock<B: Backend> {
    pub attn: MultiHeadAttention<B>,
    pub ff: FeedForward<B>,
    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
    pub drop_shortcut: Dropout,
}

#[derive(burn::prelude::Config, Copy, Debug)]
pub struct TransformerBlockConfig {
    pub context_length: usize,
    pub emb_dim: usize,
    pub nheads: usize,
    pub drop_rate: f64,
    pub qkv_bias: bool,
}

impl<B: Backend> TransformerBlock<B> {
    /// 输入维度（batch-size，num-tokens，embedding-dim）
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
}

impl TransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        let attn = MultiHeadAttentionConfig::new(
            self.emb_dim,
            self.emb_dim,
            self.context_length,
            self.drop_rate,
            self.nheads,
        )
        .with_qkv_bias(self.qkv_bias)
        .init(device);

        let ff = FeedForwardConfig::new(self.emb_dim).init(device);
        let norm1 = LayerNormConfig::new(self.emb_dim).init(device);
        let norm2 = LayerNormConfig::new(self.emb_dim).init(device);
        let drop_shortcut = Dropout { prob: self.drop_rate };

        TransformerBlock {
            attn,
            ff,
            norm1,
            norm2,
            drop_shortcut,
        }
    }
}
