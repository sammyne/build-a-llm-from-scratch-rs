use std::collections::HashSet;

use burn::backend::{Autodiff, NdArray};
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};
use chapter04::{DummyGptModel, GPT_124M};
use tiktoken::ext::Encoding;

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    let device = <B as burn::prelude::Backend>::Device::default();

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
    println!("{batch:?}");

    B::seed(123);

    let model = DummyGptModel::<B>::new(&GPT_124M, &device);
    let logits = model.forward(batch);
    println!("Output shape: {:?}", logits.shape());
    println!("logits: {logits:?}");

    Ok(())
}
