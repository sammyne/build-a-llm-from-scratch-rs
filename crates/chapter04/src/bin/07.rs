use std::collections::HashSet;

use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::Tensor;
use chapter04::{GPT_124M, GptModel};
use tiktoken::ext::Encoding;

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    B::seed(123);

    let device = <B as Backend>::Device::default();

    let batch = new_batch(&device);

    let model = GptModel::<B>::new(&GPT_124M);

    let out = model.forward(batch.clone());

    println!("Input batch:\n{batch:?}");
    println!("\nOutput shape: {:?}", out.shape());
    println!("{out}");

    // TODO: 排查和 Pytorch 不一致的原因
    let total_params = model.num_params();

    println!("Total number of parameters: {}", total_params);
    println!("Token embedding layer shape: {:?}", model.tok_emb.weight.shape());
    println!("Output layer shape: {:?}", model.out_head.weight.shape());
    println!(
        "Number of trainable parameters considering weight tying: {}",
        model.num_params() - model.out_head.num_params()
    );

    let total_size_bytes = total_params * 4;
    let total_size_mb = total_size_bytes as f32 / 1024.0 / 1024.0;
    println!("Total size of the model: {total_size_mb:.2} MB");

    Ok(())
}

fn new_batch(device: &<B as Backend>::Device) -> Tensor<B, 2, Int> {
    let tokenizer = Encoding::gpt2();

    const TXT1: &str = "Every effort moves you";
    const TXT2: &str = "Every day holds a";

    let mut batch = vec![];
    let allowed_specials = HashSet::new();
    for v in [TXT1, TXT2] {
        let ids = tokenizer.encode(v, &allowed_specials);
        let t = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), &device);
        batch.push(t);
    }

    Tensor::stack::<2>(batch, 0)
}
