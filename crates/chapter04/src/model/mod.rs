mod dummy;

use burn::module::Module;
use burn::nn::{Dropout, Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Tensor;
pub use dummy::*;

use crate::{Config, LayerNorm, LayerNormConfig, TransformerBlock, TransformerBlockConfig};

#[derive(Debug, Module)]
pub struct GptModel<B: Backend> {
    pub tok_emb: Embedding<B>,
    pub pos_emb: Embedding<B>,
    pub drop_emb: Dropout,
    pub trf_blocks: Vec<TransformerBlock<B>>,
    pub final_norm: LayerNorm<B>,
    pub out_head: Linear<B>,
}

impl<B: Backend> GptModel<B> {
    pub fn forward(&self, in_idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = in_idx.device();
        let seq_len = in_idx.shape().dims[1];

        let tok_embeds = self.tok_emb.forward(in_idx);

        let pos_embeds = self
            .pos_emb
            .forward(Tensor::arange(0..(seq_len as i64), &device).unsqueeze::<2>());

        let x = tok_embeds + pos_embeds;
        let mut x = self.drop_emb.forward(x);
        for b in &self.trf_blocks {
            x = b.forward(x);
        }
        let x = self.final_norm.forward(x);
        let logits = self.out_head.forward(x);

        logits
    }
}

impl Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GptModel<B> {
        let c = self;

        let tok_emb = EmbeddingConfig::new(c.vocab_size, c.emb_dim).init(device);
        let pos_emb = EmbeddingConfig::new(c.context_length, c.emb_dim).init(device);
        let drop_emb = Dropout { prob: c.drop_rate };

        let trf_blocks: Vec<_> = {
            let cc = TransformerBlockConfig::new(c.context_length, c.emb_dim, c.nheads, c.drop_rate, c.qkv_bias);
            (0..c.nlayers).map(|_| cc.init(device)).collect()
        };

        let final_norm = LayerNormConfig::new(c.emb_dim).init(device);
        let out_head = LinearConfig::new(c.emb_dim, c.vocab_size).with_bias(false).init(device);

        GptModel {
            tok_emb,
            pos_emb,
            drop_emb,
            trf_blocks,
            final_norm,
            out_head,
        }
    }
}
