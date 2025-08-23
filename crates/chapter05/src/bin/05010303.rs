use std::usize;

use anyhow::Context;
use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use chapter02::dataset::{self, LoaderV1Options};
use chapter02::verdict;
use chapter05::config::GPT_124M;
use chapter05::loss;
use tiktoken::ext::Encoding;

type B = Autodiff<NdArray<f32>>;
// 使用 PyTorch 后端加速明显。
// type B = Autodiff<LibTorch>;

fn main() -> anyhow::Result<()> {
    let tokenizer = Encoding::gpt2();

    let text_data = verdict::load().context("load verdict")?;

    const TRAIN_RATIO: f64 = 0.9;
    let (train_data, val_data) = {
        let i = (text_data.len() as f64 * TRAIN_RATIO) as usize;
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
        dataset::create_dataloader_v1::<B, _>(train_data, &tokenizer, opts).context("load train data")?
    };
    let val_loader = {
        let opts = LoaderV1Options {
            batch_size: 2,
            max_length: GPT_124M.context_length,
            stride: GPT_124M.context_length,
            drop_last: false,
            ..Default::default()
        };
        dataset::create_dataloader_v1::<B, _>(val_data, &tokenizer, opts).context("load validation data")?
    };

    let device = &<B as Backend>::Device::Cpu;

    let model = GPT_124M.init::<B>(&device).no_grad();

    let train_loss = loss::calc_loss_loader(train_loader.as_ref(), &model, None, &device);
    let val_loss = loss::calc_loss_loader(val_loader.as_ref(), &model, None, &device);
    println!("Train loss: {train_loss}");
    println!("Validation loss: {val_loss}");

    Ok(())
}
