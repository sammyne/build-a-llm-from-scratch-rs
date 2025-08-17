use burn::backend::{Autodiff, NdArray};
use burn::nn::Dropout;
use burn::prelude::Backend;
use burn::tensor::Tensor;

type B = Autodiff<NdArray<f32>>;

fn main() {
    let device = <B as Backend>::Device::default();

    // 3.5.2. Masking additional attention weights with dropout
    B::seed(123);
    // Dropout 的底层实现依赖 burn 的 autodiff 特性。
    let dropout = Dropout { prob: 0.5 };
    let example = Tensor::<B, 2>::ones([6, 6], &device);
    println!("{}", dropout.forward(example.clone()));
}
