mod batcher;

use std::path::Path;
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

/// 返回（训练数据集，测试数据集，验证数据集）
pub fn load_and_split<B, P, T>(
    file_path: P,
    tokenizer: &T,
) -> anyhow::Result<(
    Arc<dyn DataLoader<B, Batch<B>>>,
    Arc<dyn DataLoader<B, Batch<B>>>,
    Arc<dyn DataLoader<B, Batch<B>>>,
)>
where
    B: Backend,
    P: AsRef<Path>,
    T: Tokenizer,
{
    let (train_data, test_data, val_data) = crate::utils::load_and_split_data(file_path).context("load")?;

    const BATCH_SIZE: usize = 8;

    let mut opts = {
        let customized_collate_fn = |batch: &[Vec<u32>], device: &B::Device| {
            utils::custom_collate_fn::<B, _>(batch, None, None, Some(1024), device)
        };
        DataLoaderOptions {
            batch_size: BATCH_SIZE,
            shuffle_seed: Some(20250808),
            num_workers: 0,
            drop_last: true,
            collate_fn: customized_collate_fn,
        }
    };

    let train_dataset = InstructionDataset::new(&train_data, tokenizer).context("build train dataset")?;
    let train_loader = load(train_dataset, &opts);

    opts.shuffle_seed = None;
    opts.drop_last = false;

    let test_dataset = InstructionDataset::new(&test_data, tokenizer).context("build test dataset")?;
    let test_loader = load(test_dataset, &opts);

    let val_dataset = InstructionDataset::new(&val_data, tokenizer).context("build val dataset")?;
    let val_loader = load(val_dataset, &opts);

    Ok((train_loader, test_loader, val_loader))
}
