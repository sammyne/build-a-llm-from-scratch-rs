use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use chapter04::GPT_124M;

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    B::seed(123);

    let device = &<B as Backend>::Device::default();

    let model = GPT_124M.init::<B>(device);

    let total_params_gpt2 = model.num_params() - model.out_head.num_params();
    assert_eq!(
        124_412_160, total_params_gpt2,
        "unexpected #(trainable parameters considering weight tying)"
    );

    println!("Number of trainable parameters considering weight tying: {total_params_gpt2}");

    Ok(())
}
