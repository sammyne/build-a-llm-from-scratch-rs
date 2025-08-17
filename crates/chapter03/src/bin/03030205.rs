use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::tensor::{Tensor, activation};

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

    let attn_scores = inputs.clone().matmul(inputs.clone().transpose());

    let dim = attn_scores.dims().len() - 1;
    let attn_weights = activation::softmax(attn_scores.clone(), dim);

    // all_context_vecs[1] 和 03030105 的输出结果一致
    let all_context_vecs = attn_weights.clone().matmul(inputs.clone());
    println!("{all_context_vecs}");
}
