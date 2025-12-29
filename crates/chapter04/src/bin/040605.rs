use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use chapter04::GPT_124M;

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    let device = &<B as Backend>::Device::default();

    B::seed(device,123);


    let model = GPT_124M.init::<B>(device);

    let total_params = model.num_params();

    let total_size_bytes = total_params * 4;
    let total_size_mb = total_size_bytes as f32 / 1024.0 / 1024.0;
    println!("Total size of the model: {total_size_mb:.2} MB");

    Ok(())
}
