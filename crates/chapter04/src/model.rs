use burn::module::Module;
use burn::nn::{Dropout, Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::{Device, Int, Tensor};

use crate::{Config, DummyLayerNorm, DummyTransformerBlock, GPT_124M};

#[derive(Debug, Module)]
pub struct DummyGptModel<B: Backend> {
    tok_emb: Embedding<B>,
    pos_emb: Embedding<B>,
    drop_emb: Dropout,
    trf_blocks: Vec<DummyTransformerBlock<B>>,
    final_norm: DummyLayerNorm<B>,
    out_head: Linear<B>,
}

impl<B: Backend> DummyGptModel<B> {
    pub fn forward(&self, in_idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let seq_len = in_idx.shape().dims[1];

        let device = in_idx.device();

        let tok_embeds = self.tok_emb.forward(in_idx.clone());
        let pos_embeds = self
            .pos_emb
            .forward(Tensor::arange(0..(seq_len as i64), &device).unsqueeze::<2>());

        let x = tok_embeds + pos_embeds;
        let mut x = self.drop_emb.forward(x);
        for t in &self.trf_blocks {
            x = t.forward(x);
        }

        let x = self.final_norm.forward(x);
        self.out_head.forward(x)
    }

    pub fn new(c: &Config, d: &Device<B>) -> Self {
        let tok_emb = EmbeddingConfig::new(c.vocab_size, c.emb_dim).init(d);
        let pos_emb = EmbeddingConfig::new(c.context_length, c.emb_dim).init(d);
        let drop_emb = Dropout { prob: c.drop_rate };

        let trf_blocks: Vec<_> = (0..c.nlayers).map(|_| DummyTransformerBlock::new()).collect();
        let final_norm = DummyLayerNorm::new(c.emb_dim, None);

        let out_head = LinearConfig::new(c.emb_dim, c.vocab_size).with_bias(false).init(d);

        Self {
            tok_emb,
            pos_emb,
            drop_emb,
            trf_blocks,
            final_norm,
            out_head,
        }
    }
}
