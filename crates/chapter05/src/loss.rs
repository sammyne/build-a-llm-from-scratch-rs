use burn::data::dataloader::DataLoader;
use burn::prelude::*;
use burn::tensor::Tensor;
use chapter02::dataset::Batch;
use chapter04::GptModel;

use crate::utils;

pub fn calc_loss_batch<B: Backend>(
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

pub fn calc_loss_loader<B: Backend<FloatElem = f32>>(
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
