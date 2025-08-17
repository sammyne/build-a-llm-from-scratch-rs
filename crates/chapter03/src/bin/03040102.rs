use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::tensor::{Distribution, Tensor};

type B = NdArray<f32>;

fn main() {
    let device = <B as Backend>::Device::default();

    B::seed(123);

    let inputs = Tensor::<B, 2, _>::from_floats(
        [
            [0.43, 0.15, 0.89], // Your (x^1)
            [0.55, 0.87, 0.66], // journey (x^2)
            [0.57, 0.85, 0.64], // starts (x^3)
            [0.22, 0.58, 0.33], // with (x^4)
            [0.77, 0.25, 0.10], // one (x^5)
            [0.05, 0.80, 0.55], // step (x^6)
        ],
        &device,
    );

    let d_in = inputs.dims()[1];
    let d_out = 2;

    let distribution = Distribution::Uniform(0.0, 1.0);

    let _wq = Tensor::<B, 2_>::random([d_in, d_out], distribution, &device);
    let wk = Tensor::<B, 2_>::random([d_in, d_out], distribution, &device);
    let wv = Tensor::<B, 2_>::random([d_in, d_out], distribution, &device);

    let keys = inputs.clone().matmul(wk);
    let values = inputs.clone().matmul(wv);

    println!("keys.shape: {:?}", keys.shape());
    println!("values.shape: {:?}", values.shape());
}
