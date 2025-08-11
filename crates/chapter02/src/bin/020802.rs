use anyhow::Context;
use burn::backend::NdArray;
use burn::nn::EmbeddingConfig;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use chapter02::dataset::{self, LoaderV1Options};
use chapter02::verdict;
use tiktoken::ext::Encoding;

type B = NdArray<f32>;

fn main() -> anyhow::Result<()> {
    let device = <B as Backend>::Device::default();

    let vocab_size = 50257;
    let output_dim = 256;

    let token_embedding_layer = EmbeddingConfig::new(vocab_size, output_dim).init::<B>(&device);

    let text = verdict::load().context("load verdict")?;

    const MAX_LENGTH: usize = 4;

    let opts = LoaderV1Options {
        batch_size: 8,
        max_length: MAX_LENGTH,
        stride: MAX_LENGTH,
        shuffle_seed: None,
        ..Default::default()
    };

    let loader = dataset::create_dataloader_v1::<B, _>(&text, &Encoding::gpt2(), opts).context("new loader")?;
    let (inputs, _) = loader.iter().next().expect("fetch a data batch");

    let token_embeddings = token_embedding_layer.forward(inputs);
    println!("token-embeddings.shape = {:?}", token_embeddings.shape());

    let context_length = MAX_LENGTH;
    let pos_embedding_layer = EmbeddingConfig::new(context_length, output_dim).init::<B>(&device);

    let pos_embeddings =
        pos_embedding_layer.forward(Tensor::arange(0..(context_length as i64), &device).reshape([1, 4]));
    println!("position-embeddings.shape = {:?}", pos_embeddings.shape());

    let input_embeddings = token_embeddings + pos_embeddings;
    println!("input-embeddings.shape = {:?}", input_embeddings.shape());

    Ok(())
}
