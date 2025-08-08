mod batcher;

use std::sync::Arc;

use anyhow::Context;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::prelude::*;
use chapter02::tokenizer::Tokenizer;

pub use crate::dataset::batcher::Batch;
use crate::dataset::batcher::CollatedBatcher;
use crate::utils;

pub struct DataLoaderOptions<B: Backend> {
    pub batch_size: usize,
    pub shuffle_seed: Option<u64>,
    pub num_workers: usize,
    pub drop_last: bool,
    pub collate_fn: fn(&[Vec<u32>], &B::Device) -> Batch<B>,
}

pub struct InstructionDataset {
    encoded_texts: Vec<Vec<u32>>,
}

impl InstructionDataset {
    pub fn new<T: Tokenizer>(data: &[crate::utils::Data], tokenizer: &T) -> anyhow::Result<Self> {
        let mut encoded_texts = Vec::with_capacity(data.len());
        for entry in data {
            let instruction_plus_input = utils::format_input(entry);
            let response_text = format!("\n\n### Response:\n{}", entry.output);

            let full_text = instruction_plus_input + &response_text;

            let encoded = tokenizer
                .encode(&full_text)
                .with_context(|| format!("tokenize {full_text}"))?;
            encoded_texts.push(encoded);
        }
        Ok(Self { encoded_texts })
    }
}

impl Dataset<Vec<u32>> for InstructionDataset {
    fn len(&self) -> usize {
        self.encoded_texts.len()
    }

    fn get(&self, index: usize) -> Option<Vec<u32>> {
        self.encoded_texts.get(index).cloned()
    }
}

pub fn load<B: Backend>(
    mut dataset: InstructionDataset,
    opts: &DataLoaderOptions<B>,
) -> Arc<dyn DataLoader<B, Batch<B>>> {
    let batcher = CollatedBatcher {
        collate: opts.collate_fn,
    };

    let mut b = DataLoaderBuilder::new(batcher).batch_size(opts.batch_size);
    if opts.num_workers != 0 {
        b = b.num_workers(opts.num_workers);
    }
    if let Some(seed) = opts.shuffle_seed {
        b = b.shuffle(seed);
    }

    // 临时解决 burn 没有原生支持抛弃最后一个不完整 batch 的问题
    if opts.drop_last {
        let n = dataset.encoded_texts.len() / opts.batch_size * opts.batch_size;
        dataset.encoded_texts.truncate(n);
    }

    b.build(dataset)
}
