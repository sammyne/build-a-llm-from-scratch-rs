use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::tensor::Tensor;

type B = NdArray;

type D = <B as Backend>::Device;

fn main() {
    B::seed(123);
    let device = &D::Cpu;

    let next_token_logits =
        Tensor::<B, 1>::from_floats([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79], device);

    const TOP_K: usize = 3;

    let (top_logits, top_pos) = next_token_logits.clone().topk_with_indices(TOP_K, 0);
    println!("Top logits: {top_logits}");
    println!("Top pos: {top_pos}");
}
