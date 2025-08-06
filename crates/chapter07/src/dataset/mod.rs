mod batcher;

use anyhow::Context;
use burn::prelude::*;
use chapter02::tokenizer::Tokenizer;

use crate::utils;

pub type Data<B> = (Tensor<B, 1, Int>, Tensor<B, 1, Int>);

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

// impl<B: Backend> Dataset<Data<B>> for InstructionDataset {
//     fn len(&self) -> usize {
//         self.encoded_texts.len()
//     }

//     fn get(&self, index: usize) -> Option<Data<B>> {
//         let input = self.encoded_texts.get(index)?;
//         Some((input.clone(), target.clone()))
//     }
// }
