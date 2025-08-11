use anyhow::Context;
use burn::backend::NdArray;
use chapter02::dataset::{self, LoaderV1Options};
use chapter02::verdict;
use tiktoken::ext::Encoding;

type B = NdArray<f32>;

fn main() -> anyhow::Result<()> {
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
    println!("Token IDs:\n{inputs}");
    println!("\nInputs shape: {:?}", inputs.shape());

    Ok(())
}
