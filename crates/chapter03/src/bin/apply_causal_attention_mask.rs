use burn::backend::{Autodiff, NdArray};
use burn::nn::Dropout;
use burn::prelude::Backend;
use burn::tensor::{Tensor, activation};
use chapter03::attention::v2::SelfAttentionV2;

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
            [0.05, 0.80, 0.55],
        ], // step (x^6)
        &device,
    );

    let d_in = inputs.dims()[1];
    let d_out = 2;

    let sa_v2 = SelfAttentionV2::new(d_in, d_out, false);

    let queries = sa_v2.q.forward(inputs.clone());
    let keys = sa_v2.k.forward(inputs.clone());

    let attn_scores = queries.matmul(keys.clone().transpose());

    let dk = *keys.shape().dims.last().expect("k.shape[-1]") as f32;
    let attn_scores_last_dim = attn_scores.dims().len() - 1;
    let attn_weights = activation::softmax(attn_scores.clone() / dk.sqrt(), attn_scores_last_dim);
    println!("attn_weights:\n{attn_weights:?}\n");

    let context_length = attn_scores.shape().dims[0];
    let mask_simple = Tensor::<B, 2, _>::ones([context_length, context_length], &device).tril(0);
    println!("mask_simple:\n{mask_simple:?}\n");

    let masked_simple = attn_weights * mask_simple;
    println!("masked_simple:\n{masked_simple:?}\n");

    let dim = masked_simple.dims().len() - 1;
    let row_sums = masked_simple.clone().sum_dim(dim);
    // println!("row_sums:\n{row_sums:?}\n");

    let masked_simple_norm = masked_simple.clone() / row_sums;
    println!("masked_simple_norm:\n{masked_simple_norm:?}\n");

    // 用 -INF + softmax 做掩码
    let mask = Tensor::<B, 2>::ones([context_length, context_length], &device).triu(1);
    let masked = attn_scores.mask_fill(mask.bool(), f32::NEG_INFINITY);
    println!("masked:\n{masked:?}\n");
    let attn_weights = activation::softmax(masked.clone() / dk.sqrt(), attn_scores_last_dim);
    println!("attn_weights:\n{attn_weights:?}\n");

    // 3.5.2. Masking additional attention weights with dropout
    B::seed(123);
    // Dropout 的底层实现依赖 burn 的 autodiff 特性。
    let dropout = Dropout { prob: 0.5 };
    let example = Tensor::<B, 2>::ones([6, 6], &device);
    println!("dropout(example):\n{:?}\n", dropout.forward(example.clone()));

    B::seed(123);
    println!("dropout(attn-weights):\n{:?}\n", dropout.forward(attn_weights.clone()));
}
