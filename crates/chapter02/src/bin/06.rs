use anyhow::Context;
use chapter02::dataset::{GptDatasetV1, LoaderV1Options};
use chapter02::verdict;
use tiktoken::ext::Encoding;

type Backend = burn::backend::ndarray::NdArray<f32>;

fn main() -> anyhow::Result<()> {
    let text = verdict::load().context("load verdict")?;

    {
        let opts = LoaderV1Options {
            batch_size: 1,
            max_length: 4,
            stride: 1,
            shuffle_seed: None,
            ..Default::default()
        };

        let loader = GptDatasetV1::<Backend>::new_loader_v1(&text, &Encoding::gpt2(), opts).context("new loader")?;
        let mut iter = loader.iter();

        let b1 = iter.next().context("get 1st batch")?;
        println!("{b1:?}");

        let b2 = iter.next().context("get 2nd batch")?;
        println!("{b2:?}");
    }

    // batch size > 1
    {
        let opts = LoaderV1Options {
            batch_size: 8,
            max_length: 4,
            stride: 4,
            shuffle_seed: None,
            ..Default::default()
        };
        let loader = GptDatasetV1::<Backend>::new_loader_v1(&text, &Encoding::gpt2(), opts).context("new loader")?;
        let mut iter = loader.iter();

        let (inputs, targets) = iter.next().context("get 1st batch")?;
        println!("{inputs:?}");
        println!("{targets:?}");
    }

    Ok(())
}
