use std::collections::HashSet;

use burn::backend::{Autodiff, NdArray};
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};
use tiktoken::ext::Encoding;

type B = Autodiff<NdArray<f32>>;

fn main() {
    let device = <B as Backend>::Device::default();

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

    let batch = Tensor::stack::<2>(batch, 0);
    println!("{batch}");
}
