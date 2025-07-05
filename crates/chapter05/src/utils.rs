use std::collections::HashSet;

use anyhow::Context as _;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::tensor::DType;
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
