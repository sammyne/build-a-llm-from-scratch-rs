use std::collections::HashSet;

use anyhow::Context as _;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::tensor::{DType, activation};
use chapter04::GptModel;
use tiktoken::ext::Encoding;

pub trait Tokenizer<B: Backend> {
    fn detokenize(&self, ids: Tensor<B, 2, Int>) -> anyhow::Result<String>;

    fn tokenize(&self, text: &str) -> Tensor<B, 2, Int>;
}

impl<B: Backend> Tokenizer<B> for Encoding {
    fn detokenize(&self, ids: Tensor<B, 2, Int>) -> anyhow::Result<String> {
        let ids: Vec<u32> = ids
            .squeeze::<1>(0)
            .to_data()
            .convert_dtype(DType::U32)
            .into_vec()
            .map_err(|err| anyhow::anyhow!("conv out ids: {err:?}"))?;

        //self.decode_str(&ids).context("decode output ids")
        let b = self.decode(&ids).context("decode out ids")?;
        Ok(String::from_utf8_lossy(&b).to_string())
    }

    fn tokenize(&self, text: &str) -> Tensor<B, 2, Int> {
        let device = B::Device::default();

        let allowed_specials = HashSet::<&'static str>::from(["<|endoftext|>"]);
        let encoded = self.encode(text, &allowed_specials);

        Tensor::<B, 1, Int>::from_ints(encoded.as_slice(), &device).unsqueeze::<2>()
    }
}

pub fn cross_entropy<B: Backend, const D: usize, const D2: usize>(
    logits: Tensor<B, D>,
    target_indices: Tensor<B, D2, Int>,
) -> Tensor<B, 1> {
    assert_eq!(D, D2 + 1, "target_indices must be one dimension less than logits");

    // let p = activation::log_softmax(logits, D2);
    // println!("p: {}", p);

    // let indices = target_indices.unsqueeze_dim::<D>(D2);

    // // let p = p.gather(D2, indices);
    // // println!("p: {}", p);

    // // let p = p.flatten::<1>(0, D2).neg().mean();
    // // println!("p: {}", p);

    // activation::log_softmax(logits, D2)
    //     .gather(D2, indices)
    //     .flatten::<1>(0, D2)
    //     .neg()
    //     .mean()

    let logits = logits.flatten::<2>(0, D2 - 1);
    let target_indices = target_indices.flatten::<1>(0, D2 - 1);

    CrossEntropyLossConfig::new()
        .init::<B>(&logits.device())
        .forward(logits, target_indices)
}

/// TODO: 将 eos_id 的类型调整为 Option<u32>
pub fn generate<B: Backend<IntElem = i64>>(
    model: &GptModel<B>,
    mut idx: Tensor<B, 2, Int>,
    max_new_tokens: usize,
    context_size: usize,
    temperature: Option<f32>,
    topk: Option<usize>,
    eos_id: Option<usize>,
) -> Tensor<B, 2, Int> {
    let context_size = context_size as i32;

    for _ in 0..max_new_tokens {
        let idx_cond = idx.clone().slice(s![.., -context_size..]);

        let logits = model.forward(idx_cond);
        let mut logits = logits.slice(s![.., -1, ..]).squeeze(1);

        if let Some(k) = topk {
            let top_logits = logits.clone().squeeze::<1>(0).topk(k, 0);
            let v = top_logits.min().into_scalar();
            let discarded = logits.clone().lower(logits.clone().full_like(v));
            logits = logits.mask_fill(discarded, f32::NEG_INFINITY);
        }
        // println!("probas.shape: {:?}", logits.shape());

        let dim = logits.dims().len() - 1;
        let idx_next = match temperature {
            Some(t) => {
                logits = logits / t;
                let probas = activation::softmax(logits.clone(), dim);
                // 自己实现的 multinomial 目前看起来太吃内存，导致 OOM。
                crate::rand::multinomial(probas)
            }
            None => logits.argmax(dim),
        };

        match eos_id {
            Some(v) if idx_next.clone().squeeze::<1>(0).into_scalar() == v as i64 => break,
            _ => {}
        }

        idx = Tensor::cat(vec![idx, idx_next], 1);
    }

    idx
}
