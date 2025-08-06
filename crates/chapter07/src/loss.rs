use burn::config::Config;
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};

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
