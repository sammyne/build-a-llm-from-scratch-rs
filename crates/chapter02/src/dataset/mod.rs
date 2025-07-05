mod internal;

use std::sync::Arc;

use anyhow::Context;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::prelude::{Backend, Tensor};
use burn::tensor::Int;

pub use crate::dataset::internal::Batch;
use crate::tokenizer::Tokenizer;

pub type Data<B> = (Tensor<B, 1, Int>, Tensor<B, 1, Int>);

pub struct GptDatasetV1<B: Backend> {
    input_ids: Vec<Tensor<B, 1, Int>>,
    target_ids: Vec<Tensor<B, 1, Int>>,
}

pub struct LoaderV1Options {
    pub batch_size: usize,
    pub max_length: usize,
    pub stride: usize,
    pub shuffle_seed: Option<u64>,
    /// TODO(xiangminli): 使用这个字段
    pub drop_last: bool,
    pub num_workers: usize,
}

impl<B: Backend> GptDatasetV1<B> {
    pub fn new<T: Tokenizer>(text: &str, tokenizer: &T, max_length: usize, stride: usize) -> anyhow::Result<Self> {
        let token_ids = tokenizer.encode(text).context("tokenize")?;

        let mut input_ids = vec![];
        let mut target_ids = vec![];

        let device = B::Device::default();
        for i in (0..(token_ids.len() - max_length)).step_by(stride) {
            let input = Tensor::from_ints(&token_ids[i..(i + max_length)], &device);
            let target = Tensor::from_ints(&token_ids[(i + 1)..(i + 1 + max_length)], &device);
            input_ids.push(input);
            target_ids.push(target);
        }

        let out = Self { input_ids, target_ids };
        Ok(out)
    }

    pub fn new_loader_v1<T: Tokenizer>(
        text: &str,
        tokenizer: &T,
        opts: LoaderV1Options,
    ) -> anyhow::Result<Arc<dyn DataLoader<B, Batch<B>>>> {
        let mut b = DataLoaderBuilder::new(internal::Batcher::default()).batch_size(opts.batch_size);
        if opts.num_workers != 0 {
            b = b.num_workers(opts.num_workers);
        }
        if let Some(seed) = opts.shuffle_seed {
            b = b.shuffle(seed);
        }

        let dataset = Self::new(text, tokenizer, opts.max_length, opts.stride).context("new dataset")?;

        let out = b.build(dataset);

        Ok(out)
    }
}

impl<B: Backend> Dataset<Data<B>> for GptDatasetV1<B> {
    fn len(&self) -> usize {
        self.input_ids.len()
    }

    fn get(&self, index: usize) -> Option<Data<B>> {
        let input = self.input_ids.get(index)?;
        let target = self.target_ids.get(index)?;
        Some((input.clone(), target.clone()))
    }
}

impl Default for LoaderV1Options {
    fn default() -> Self {
        Self {
            batch_size: 4,
            max_length: 256,
            stride: 128,
            shuffle_seed: Some(123),
            drop_last: true,
            num_workers: 0,
        }
    }
}
