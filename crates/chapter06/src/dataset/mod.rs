mod batcher;

use std::ops::Deref;

use anyhow::Context as _;
pub use batcher::{Batch, Batcher};
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::prelude::*;
use chapter02::tokenizer::Tokenizer;
use polars::prelude::*;

pub type Data<B> = (Tensor<B, 1, Int>, Tensor<B, 1, Int>);

#[derive(Config)]
pub struct DataLoaderOptions {
    #[config(default = 8)]
    pub batch_size: usize,
    pub shuffle_seed: Option<u64>,
    #[config(default = 0)]
    pub num_workers: usize,
    #[config(default = false)]
    pub drop_last: bool,
}

pub struct LoadCsvOptions<'a, D, T> {
    path: &'a str,
    tokenizer: &'a T,
    max_length: Option<usize>,
    pad_token_id: u32,
    device: &'a D,
}

pub struct SpamDataset<B: Backend> {
    pub encoded_texts: Vec<Tensor<B, 1, Int>>,
    pub labels: Vec<Tensor<B, 1, Int>>,
    pub max_length: usize,
}

impl<'a, D, T> LoadCsvOptions<'a, D, T> {
    // pub fn with_max_length(mut self, v: usize) -> Self {
    //     self.max_length = Some(v);
    //     self
    // }

    // pub fn with_pad_token_id(mut self, v: u32) -> Self {
    //     self.pad_token_id = v;
    //     self
    // }

    pub fn new(path: &'a str, tokenizer: &'a T, device: &'a D) -> Self {
        Self {
            path,
            tokenizer,
            max_length: None,
            pad_token_id: 50256,
            device,
        }
    }
}

impl<B: Backend> SpamDataset<B> {
    pub fn load_csv<T: Tokenizer>(opts: LoadCsvOptions<'_, B::Device, T>) -> anyhow::Result<Self> {
        let LoadCsvOptions {
            path,
            tokenizer,
            max_length,
            pad_token_id,
            device,
        } = opts;

        let data = crate::utils::load_csv(path).context("load csv")?;

        let (encoded_texts, max_length) = {
            let c = match &data["Text"] {
                Column::Series(c) => c.deref(),
                _ => anyhow::bail!("expect series column for texts"),
            };

            let mut encoded_texts = Vec::with_capacity(c.len());
            for v in c.rechunk().iter() {
                let v = match v {
                    AnyValue::String(v) => v,
                    unexpected => anyhow::bail!("unexpected non-string text: {unexpected:?}"),
                };

                let encoded = tokenizer.encode(v).with_context(|| format!("tokenize '{v}'"))?;
                encoded_texts.push(encoded);
            }

            let max_length = match max_length {
                Some(v) => v,
                None => encoded_texts
                    .iter()
                    .map(|v| v.len())
                    .max()
                    .context("cannot infer max length")?,
            };

            let mut encoded_text_tensors = Vec::with_capacity(encoded_texts.len());

            for v in encoded_texts.iter_mut() {
                v.resize(max_length, pad_token_id);

                encoded_text_tensors.push(Tensor::from_ints(v.as_slice(), device));
            }

            (encoded_text_tensors, max_length)
        };

        let labels = {
            let c = match &data["Label"] {
                Column::Series(c) => c.deref(),
                _ => anyhow::bail!("expect series column for labels"),
            };

            let mut out = Vec::with_capacity(c.len());
            for v in c.rechunk().iter() {
                let v = match v {
                    AnyValue::UInt8(v) => v,
                    AnyValue::UInt16(v) => v as u8,
                    AnyValue::UInt32(v) => v as u8,
                    AnyValue::UInt64(v) => v as u8,
                    AnyValue::Int8(v) => v as u8,
                    AnyValue::Int16(v) => v as u8,
                    AnyValue::Int32(v) => v as u8,
                    AnyValue::Int64(v) => v as u8,
                    unexpected => anyhow::bail!("unexpected label: {unexpected:?}"),
                };
                out.push(Tensor::from_ints([v], device));
            }

            out
        };

        let out = Self {
            encoded_texts,
            labels,
            max_length,
        };

        Ok(out)
    }
}

impl<B: Backend> Dataset<Data<B>> for SpamDataset<B> {
    fn len(&self) -> usize {
        self.encoded_texts.len()
    }

    fn get(&self, index: usize) -> Option<Data<B>> {
        let input = self.encoded_texts.get(index)?;
        let target = self.labels.get(index)?;
        Some((input.clone(), target.clone()))
    }
}

pub fn load<B: Backend>(mut dataset: SpamDataset<B>, opts: DataLoaderOptions) -> Arc<dyn DataLoader<B, Batch<B>>> {
    let mut b = DataLoaderBuilder::new(Batcher::default()).batch_size(opts.batch_size);
    if opts.num_workers != 0 {
        b = b.num_workers(opts.num_workers);
    }
    if let Some(seed) = opts.shuffle_seed {
        b = b.shuffle(seed);
    }

    // 临时解决 burn 没有原生支持抛弃最后一个不完整 batch 的问题
    if opts.drop_last {
        let n = dataset.labels.len() / opts.batch_size * opts.batch_size;
        dataset.encoded_texts.truncate(n);
        dataset.labels.truncate(n);
    }

    b.build(dataset)
}
