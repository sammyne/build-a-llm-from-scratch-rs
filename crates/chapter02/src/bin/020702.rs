use burn::backend::NdArray;
use burn::nn::EmbeddingConfig;
use burn::prelude::Backend;
use burn::tensor::Tensor;

type B = NdArray<f32>;

fn main() {
    B::seed(123);

    let device = <B as Backend>::Device::default();

    let vocab_size = 6;
    let output_dim = 3;

    let embedding = EmbeddingConfig::new(vocab_size, output_dim).init::<B>(&device);

    let out = embedding.forward(Tensor::from_ints([[3]], &device));
    println!("{out}");
}
