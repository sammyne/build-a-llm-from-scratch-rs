use burn::prelude::*;

pub type Batch<B> = (Tensor<B, 2, Int>, Tensor<B, 2, Int>);

pub struct CollatedBatcher<B: Backend> {
    pub collate: fn(&[Vec<u32>], device: &B::Device) -> Batch<B>,
}

impl<B: Backend> burn::data::dataloader::batcher::Batcher<B, Vec<u32>, Batch<B>> for CollatedBatcher<B> {
    fn batch(&self, items: Vec<Vec<u32>>, device: &<B as Backend>::Device) -> Batch<B> {
        (self.collate)(&items, device)
    }
}
