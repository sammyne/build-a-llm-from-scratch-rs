use std::fmt::Display;
use std::fs::File;
use std::ops::Deref;
use std::path::Path;

use anyhow::Context as _;
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::PAD_TOKEN_ID;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Data {
    pub instruction: String,
    pub input: Option<String>,
    pub output: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataWithModelResponse {
    #[serde(flatten)]
    pub data: Data,
    pub model_response: String,
}

impl Display for Data {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string(self).expect("serde-json::to_string");
        write!(f, "{json}")
    }
}

impl Deref for DataWithModelResponse {
    type Target = Data;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Listing 7.5 Implementing a custom batch collate function
pub fn custom_collate_fn<B: Backend, T: AsRef<[u32]>>(
    batch: &[T],
    pad_token_id: Option<u32>,
    ignored_index: Option<i32>,
    allowed_max_length: Option<usize>,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let batch_max_length = batch.iter().map(|x| x.as_ref().len()).max().unwrap_or(0) + 1;
    let pad_token_id = pad_token_id.unwrap_or(PAD_TOKEN_ID);
    let ignored_index = ignored_index.unwrap_or(-100);

    let mut inputs_lst = Vec::with_capacity(batch.len());
    let mut targets_lst = Vec::with_capacity(batch.len());
    for item in batch {
        let mut padded = item.as_ref().to_vec();

        padded.resize(batch_max_length, pad_token_id);
        let mut inputs = Tensor::<B, 1, Int>::from_ints(&padded[..batch_max_length - 1], device);

        // 将第一个 pad_token_id 往后的所有 target 元素置为 ignored_index。
        let mut targets = {
            let p = &padded[1..];
            let mut v = p.iter().map(|v| *v as i32).collect::<Vec<_>>();
            for (a, b) in v.iter_mut().rev().zip(p.iter().rev().skip(1)) {
                if b != &pad_token_id {
                    break;
                }
                *a = ignored_index;
            }

            Tensor::<B, 1, Int>::from_ints(v.as_slice(), device)
        };

        if let Some(n) = allowed_max_length {
            inputs = inputs.slice(0..n);
            targets = targets.slice(0..n);
        }

        inputs_lst.push(inputs);
        targets_lst.push(targets);
    }

    (Tensor::stack(inputs_lst, 0), Tensor::stack(targets_lst, 0))
}

/// Listing 7.2 Implementing the prompt formatting function
pub fn format_input(entry: &Data) -> String {
    let instruction_text = format!(
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\
        \n\n### Instruction:\n{}",
        entry.instruction
    );

    let input_text = match &entry.input {
        Some(v) if !v.is_empty() => format!("\n\n### Input:\n{v}"),
        _ => "".to_owned(),
    };

    instruction_text + &input_text
}

pub fn load_json<P, T>(file_path: P) -> anyhow::Result<Vec<T>>
where
    P: AsRef<Path>,
    T: DeserializeOwned,
{
    let f = File::open(file_path.as_ref()).context("open file")?;

    let out = serde_json::from_reader(f).context("json decode")?;

    Ok(out)
}

pub fn load_and_split_data<P, T>(file_path: P) -> anyhow::Result<(Vec<T>, Vec<T>, Vec<T>)>
where
    P: AsRef<Path>,
    T: Clone + DeserializeOwned,
{
    let data = load_json(file_path).context("load file")?;

    let train_portion = (data.len() as f32 * 0.85) as usize;
    let test_portion = (data.len() as f32 * 0.1) as usize;

    let train_data = &data[..train_portion];
    let test_data = &data[train_portion..][..test_portion];
    let val_data = &data[(train_portion + test_portion)..];

    Ok((train_data.to_vec(), test_data.to_vec(), val_data.to_vec()))
}
