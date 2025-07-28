use std::sync::Arc;

use burn::data::dataloader::DataLoader;
use burn::module::AutodiffModule as _;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::s;
use chapter04::GptModel;

use crate::dataset::Batch;

/// Listing 6.8 Calculating the classification accuracy
pub fn calc_accuracy_loader<B: AutodiffBackend>(
    data_loader: Arc<dyn DataLoader<B, Batch<B>>>,
    model: &GptModel<B>,
    device: &B::Device,
    num_batches: Option<usize>,
) -> f32 {
    let model = model.valid();
    let mut correct_predictions = 0;
    let mut num_examples = 0;

    let data_len = data_loader.iter().count();
    let num_batches = match num_batches {
        None => data_len,
        Some(v) => v.min(data_len),
    };

    for (input_batch, target_batch) in data_loader.iter().take(num_batches) {
        let input_batch = input_batch.to_device(device).valid();
        let target_batch = target_batch.to_device(device).valid();

        let logits = model.forward(input_batch).slice(s![.., -1, ..]).squeeze_dims::<2>(&[1]);
        let dim = logits.dims().len() - 1;
        let predicted_labels = logits.argmax(dim);

        num_examples += predicted_labels.dims()[0];
        correct_predictions += predicted_labels
            .equal(target_batch)
            .int()
            .sum()
            .into_data()
            .to_vec::<i64>()
            .expect("tensor as Vec<i64>")[0] as u32;
    }

    correct_predictions as f32 / num_examples as f32
}
