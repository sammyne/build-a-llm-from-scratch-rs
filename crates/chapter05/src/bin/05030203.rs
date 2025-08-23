use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::tensor::{Tensor, activation};

type B = NdArray;

type D = <B as Backend>::Device;

fn main() {
    B::seed(123);
    let device = &D::Cpu;

    let next_token_logits =
        Tensor::<B, 1>::from_floats([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79], device);

    const TOP_K: usize = 3;

    let top_logits = next_token_logits.clone().topk(TOP_K, 0);

    let new_logits = {
        let v = top_logits.min().into_scalar();
        let discarded = next_token_logits.clone().lower(next_token_logits.clone().full_like(v));
        next_token_logits.clone().mask_fill(discarded, f32::NEG_INFINITY)
    };

    let topk_probas = activation::softmax(new_logits, 0);
    // TODO(xiangminli): 调研是否能够格式化输出的 f32，保持和书上的一样
    println!("Topk probas: {topk_probas}");
}
