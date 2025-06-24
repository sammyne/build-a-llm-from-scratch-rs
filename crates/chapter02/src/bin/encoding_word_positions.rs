use anyhow::Context;
use burn::nn::EmbeddingConfig;
use burn::tensor::Tensor;
use chapter02::dataset::{GptDatasetV1, LoaderV1Options};
use chapter02::verdict;
use tiktoken::ext::Encoding;

type B = burn::backend::ndarray::NdArray<f32>;

fn main() -> anyhow::Result<()> {
    let device = <B as burn::prelude::Backend>::Device::default();

    let vocab_size = 50257;
    let output_dim = 256;

    let token_embedding_layer = EmbeddingConfig::new(vocab_size, output_dim).init::<B>(&device);

    let text = verdict::load().context("load verdict")?;

    let max_length = 4;

    let opts = LoaderV1Options {
        batch_size: 8,
        max_length,
        stride: max_length,
        shuffle_seed: None,
        ..Default::default()
    };

    let loader = GptDatasetV1::<B>::new_loader_v1(&text, Encoding::gpt2(), opts).context("new loader")?;
    let (inputs, _) = loader.iter().next().expect("fetch a data batch");
    println!("Token IDs:\n{inputs}");
    println!("\nInputs shape: {:?}", inputs.shape());

    let token_embeddings = token_embedding_layer.forward(inputs);
    println!("token-embeddings.shape = {:?}", token_embeddings.shape());

    let context_length = max_length;
    let pos_embedding_layer = EmbeddingConfig::new(context_length, output_dim).init::<B>(&device);
    println!("{:?}", Tensor::<B, 1, _>::arange(0..(context_length as i64), &device));
    let pos_embeddings =
        pos_embedding_layer.forward(Tensor::arange(0..(context_length as i64), &device).reshape([1, 4]));
    println!("position-embeddings.shape = {:?}", pos_embeddings.shape());

    let input_embeddings = token_embeddings + pos_embeddings;
    println!("input-embeddings.shape = {:?}", input_embeddings.shape());

    Ok(())
}
