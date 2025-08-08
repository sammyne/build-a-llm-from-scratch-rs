use burn::config::Config;
use burn::data::dataloader::DataLoader;
use burn::prelude::*;
use burn::tensor::{Int, Tensor};
use chapter02::dataset::Batch;
use chapter04::GptModel;

pub struct CrossEntropyLoss {
    pub ignore_index: i32,
}

#[derive(Config)]
pub struct CrossEntropyLossConfig {
    /// 忽略的目标索引。
    #[config(default = -100)]
    pub ignore_index: i32,
}

impl CrossEntropyLoss {
    pub fn forward<B: Backend>(&self, logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Tensor<B, 1> {
        let (logits, targets) = self.discard_ignored(logits, targets);

        burn::nn::loss::CrossEntropyLossConfig::new()
            .init::<B>(&logits.device())
            .forward(logits, targets)
    }

    pub fn discard_ignored<B: Backend>(
        &self,
        logits: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> (Tensor<B, 2>, Tensor<B, 1, Int>) {
        let ignored = targets
            .clone()
            .equal_elem(self.ignore_index)
            .into_data()
            .into_vec::<bool>()
            .expect("bool tensor as bool vec");
        if !ignored.iter().any(|&v| v) {
            return (logits, targets);
        }

        let kept = ignored
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { None } else { Some(i) })
            .collect::<Vec<_>>();
        let kept = Tensor::<B, 1, Int>::from_ints(kept.as_slice(), &logits.device());

        let logits = logits.select(0, kept.clone());
        let targets = targets.select(0, kept);

        (logits, targets)
    }
}

impl CrossEntropyLossConfig {
    pub fn init(&self) -> CrossEntropyLoss {
        CrossEntropyLoss {
            ignore_index: self.ignore_index,
        }
    }
}

pub fn calc_loss_batch<B: Backend>(
    input_batch: Tensor<B, 2, Int>,
    target_batch: Tensor<B, 2, Int>,
    model: &GptModel<B>,
    device: &<B as Backend>::Device,
) -> Tensor<B, 1> {
    let input_batch = input_batch.to_device(&device);
    let target_batch = target_batch.to_device(&device);

    let logits = model.forward(input_batch);

    let logits = logits.flatten::<2>(0, 1);
    let targets = target_batch.flatten::<1>(0, 1);

    CrossEntropyLossConfig::new().init().forward(logits, targets)
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
