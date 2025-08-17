use burn::backend::{Autodiff, NdArray};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use chapter03::attention::MultiHeadAttentionWrapper;

type B = Autodiff<NdArray<f32>>;

fn main() {
    let device = <B as burn::prelude::Backend>::Device::default();

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

    let batch = Tensor::stack::<3>(vec![inputs.clone(), inputs.clone()], 0);

    let context_length = batch.shape().dims[1];
    let mha = MultiHeadAttentionWrapper::new(d_in, d_out, context_length, 0.0, 2, false);
    let context_vecs = mha.forward(batch);
    println!("{context_vecs}");
    println!("\ncontext_vecs.shape: {:?}\n", context_vecs.shape());
}
