use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use chapter03::attention::SelfAttentionV1;

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

    B::seed(123);
    let sa_v1 = SelfAttentionV1::<B>::new(d_in, d_out);
    let context_vecs = sa_v1.forward(inputs);
    println!("{context_vecs}");
}
