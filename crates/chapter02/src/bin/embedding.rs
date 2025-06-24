use burn::backend::NdArray;
use burn::nn::EmbeddingConfig;
use burn::prelude::Backend as _;
use burn::tensor::Tensor;

type Backend = burn::backend::ndarray::NdArray<f32>;

fn main() {
    Backend::seed(123);

    let device = <NdArray as burn::prelude::Backend>::Device::default();

    let vocab_size = 6;
    let output_dim = 3;

    let embedding = EmbeddingConfig::new(vocab_size, output_dim).init::<Backend>(&device);
    println!("Parameter containing:");
    println!("{:?}", embedding.weight.val());

    let out = embedding.forward(Tensor::from_ints([[3]], &device));
    println!("\n{out:?}");

    let input_ids = Tensor::<Backend, 1, _>::from_ints([2, 3, 5, 1], &device).reshape([4, 1]);
    let out = embedding.forward(input_ids);
    println!("\n{out:?}");
}
