use burn::backend::NdArray;
use burn::nn::EmbeddingConfig;
use burn::prelude::Backend;

type B = NdArray<f32>;

fn main() {
    B::seed(123);

    let device = <B as Backend>::Device::default();

    let vocab_size = 6;
    let output_dim = 3;

    let embedding = EmbeddingConfig::new(vocab_size, output_dim).init::<B>(&device);
    println!("Parameter containing:");
    println!("{:?}", embedding.weight.val());
}
