use anyhow::Context;
use chapter02::dataset::{self, LoaderV1Options};
use chapter02::verdict;
use tiktoken::ext::Encoding;

type B = burn::backend::ndarray::NdArray<f32>;

fn main() -> anyhow::Result<()> {
    let text = verdict::load().context("load verdict")?;

    let opts = LoaderV1Options {
        batch_size: 8,
        max_length: 4,
        stride: 4,
        shuffle_seed: None,
        ..Default::default()
    };

    let loader = dataset::create_dataloader_v1::<B, _>(&text, &Encoding::gpt2(), opts).context("new loader")?;
    let mut iter = loader.iter();

    let (inputs, targets) = iter.next().context("get 1st batch")?;
    println!("Inputs:\n{inputs}");
    println!("\nTargets:\n{targets}");

    Ok(())
}
