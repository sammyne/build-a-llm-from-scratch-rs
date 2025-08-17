use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::tensor::Tensor;

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

    let d0 = inputs.shape().dims[0];

    let mut attn_scores = Tensor::<B, 2>::zeros([d0, d0], &device);
    for (i, x) in inputs.clone().iter_dim(0).enumerate() {
        for (j, y) in inputs.clone().iter_dim(0).enumerate() {
            let z = x.clone().mul(y).sum().unsqueeze();

            attn_scores = attn_scores.slice_assign([i..(i + 1), j..(j + 1)], z);
        }
    }
    println!("{attn_scores}\n");
}
