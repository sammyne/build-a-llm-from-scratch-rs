use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};

pub trait TensorExt<B: Backend> {
    fn cumsum(self) -> Self;

    fn search_sorted(self, values: Tensor<B, 1>) -> Tensor<B, 1, Int>;
}

impl<B: Backend> TensorExt<B> for Tensor<B, 2> {
    fn cumsum(self) -> Self {
        let device = &self.device();

        let [_, d1] = self.dims();
        let v = Tensor::<B, 2>::ones([d1, d1], device).triu(0);

        self.matmul(v)
    }

    fn search_sorted(self, values: Tensor<B, 1>) -> Tensor<B, 1, Int> {
        let device = &self.device();

        let [d0, d1] = self.dims();

        let values = values.unsqueeze::<2>().transpose().expand([d0, d1]);

        let discarded = self.lower_equal(values);

        let indices = Tensor::<B, 1, Int>::arange(0..(d1 as i64), device).unsqueeze::<2>();

        let indices = indices.expand([d0, d1]);

        let indices = indices.mask_fill(discarded, d1 as i64);

        let out = indices.min_dim(1).transpose().squeeze::<1>(0);

        out
    }
}
