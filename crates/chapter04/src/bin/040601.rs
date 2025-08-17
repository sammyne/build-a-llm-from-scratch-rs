use std::collections::HashSet;

use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::Tensor;
use chapter04::GPT_124M;
use tiktoken::ext::Encoding;

type B = Autodiff<NdArray<f32>>;

fn main() {
    B::seed(123);

    let device = &<B as Backend>::Device::default();

    let batch = new_batch(device);

    let model = GPT_124M.init(device);

    let out = model.forward(batch.clone());

    println!("Input batch:\n{batch}");
    println!("\nOutput shape: {:?}", out.shape());
    println!("{out}");
}

fn new_batch(device: &<B as Backend>::Device) -> Tensor<B, 2, Int> {
    let tokenizer = Encoding::gpt2();

    const TXT1: &str = "Every effort moves you";
    const TXT2: &str = "Every day holds a";

    let mut batch = vec![];
    let allowed_specials = HashSet::new();
    for v in [TXT1, TXT2] {
        let ids = tokenizer.encode(v, &allowed_specials);
        let t = Tensor::<B, 1, Int>::from_ints(ids.as_slice(), device);
        batch.push(t);
    }

    Tensor::stack::<2>(batch, 0)
}
