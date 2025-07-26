use std::marker::PhantomData;

use burn::prelude::*;

use crate::dataset::Data;

pub type Batch<B> = (Tensor<B, 2, Int>, Tensor<B, 2, Int>);

#[derive(Default)]
pub struct Batcher<B: Backend> {
    _p: PhantomData<B>,
}

impl<B: Backend> burn::data::dataloader::batcher::Batcher<B, Data<B>, Batch<B>> for Batcher<B> {
    fn batch(&self, items: Vec<Data<B>>, _device: &<B as Backend>::Device) -> Batch<B> {
        let (inputs, targets): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        let inputs = Tensor::stack(inputs, 0);
        let targets = Tensor::stack(targets, 0);
        (inputs, targets)
    }
}

