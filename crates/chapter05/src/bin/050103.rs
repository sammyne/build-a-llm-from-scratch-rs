use std::usize;

use anyhow::Context;
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{Autodiff, LibTorch};
use burn::data::dataloader::DataLoader;
use burn::prelude::*;
use chapter02::dataset::{Batch, GptDatasetV1, LoaderV1Options};
use chapter02::verdict;
use chapter04::GptModel;
use chapter05::config::GPT_124M;
use chapter05::utils::{self, Tokenizer};
use tiktoken::ext::Encoding;

// type B = Autodiff<NdArray<f32>>;
// 使用 PyTorch 后端加速明显。
type B = Autodiff<LibTorch>;

fn main() -> anyhow::Result<()> {
    let tokenizer = Encoding::gpt2();

    let text_data = verdict::load().context("load verdict")?;

    let tokens: Tensor<B, 2, Int> = tokenizer.tokenize(&text_data);
    let total_tokens: usize = tokens.shape().dims.iter().product();
    println!("Characters: {}", text_data.len());
    println!("Tokens: {total_tokens}");

    // let model = GptModel::<B>::new(GPT_124M).no_grad();

    const TRAIN_RATIO: f64 = 0.9;
    let (train_data, val_data) = {
        let i = (text_data.len() as f64 * TRAIN_RATIO) as usize;
        println!("{i}");
        text_data.split_at(i)
    };

    B::seed(123);

    let train_loader = {
        let opts = LoaderV1Options {
            batch_size: 2,
            max_length: GPT_124M.context_length,
            stride: GPT_124M.context_length,
            drop_last: true,
            shuffle_seed: Some(123),
            ..Default::default()
        };
        GptDatasetV1::<B>::new_loader_v1(train_data, &tokenizer, opts).context("load train data")?
    };
    let val_loader = {
        let opts = LoaderV1Options {
            batch_size: 2,
            max_length: GPT_124M.context_length,
            stride: GPT_124M.context_length,
            drop_last: false,
            ..Default::default()
        };
        GptDatasetV1::<B>::new_loader_v1(val_data, &tokenizer, opts).context("load validation data")?
    };

    println!("Train loader:");
    for (x, y) in train_loader.iter() {
        println!("shape(x,y)=({:?},{:?})", x.shape(), y.shape());
    }
    println!("\nValidation loader:");
    for (x, y) in val_loader.iter() {
        println!("shape(x,y)=({:?},{:?})", x.shape(), y.shape());
    }

    let device = LibTorchDevice::Mps;

    let model = GptModel::<B>::new(GPT_124M).no_grad();
    let model = model.to_device(&device);

    let train_loss = calc_loss_loader(train_loader.as_ref(), &model, None, &device);
    let val_loss = calc_loss_loader(train_loader.as_ref(), &model, None, &device);
    println!("Train loss: {train_loss}");
    println!("Validation loss: {val_loss}");

    Ok(())
}

fn calc_loss_batch(
    input_batch: Tensor<B, 2, Int>,
    target_batch: Tensor<B, 2, Int>,
    model: &GptModel<B>,
    device: &<B as Backend>::Device,
) -> Tensor<B, 1> {
    let input_batch = input_batch.to_device(&device);
    let target_batch = target_batch.to_device(&device);

    let logits = model.forward(input_batch);
    utils::cross_entropy(logits, target_batch)
}

fn calc_loss_loader(
    data_loader: &dyn DataLoader<B, Batch<B>>,
    model: &GptModel<B>,
    nbatches: Option<usize>,
    device: &<B as Backend>::Device,
) -> f32 {
    let mut total_loss = 0.0;
    if data_loader.num_items() == 0 {
        return f32::NAN;
    }

    let nbatches = nbatches.unwrap_or(usize::MAX);
    let mut n = 0;
    for (input_batch, target_batch) in data_loader.iter().take(nbatches) {
        let loss = calc_loss_batch(input_batch, target_batch, &model, device);
        // println!("total-loss={total_loss}, loss={loss}");
        total_loss += loss.into_scalar();
        n += 1;
    }

    total_loss / (n as f32)
}
