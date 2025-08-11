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

    let input_ids = Tensor::<B, 1, _>::from_ints([2, 3, 5, 1], &device).reshape([4, 1]);
    let out = embedding.forward(input_ids);
    println!("\n{out}");
}
