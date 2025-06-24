use burn::tensor::{Int, Tensor, activation};

type B = burn::backend::ndarray::NdArray<f32>;

fn main() {
    let device = <B as burn::prelude::Backend>::Device::default();

    // 3.3.1. A simple self-attention mechanism without trainable weights
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

    let i = Tensor::<B, 1, Int>::from_data([1], &device);
    let query = inputs.clone().select(0, i).reshape([3]);
    println!("{query:?}");

    let d0 = inputs.shape().dims[0];
    let d1 = inputs.shape().dims[1];

    let mut attn_score = Tensor::<B, 1>::empty([d0], &device);
    for i in 0..d0 {
        let x = inputs.clone().select(0, [i].into()).reshape([d1]);
        attn_score = attn_score.slice_assign([i..(i + 1)], x.mul(query.clone()).sum());
    }
    println!("attn-score\n{attn_score:?}");

    let attn_weights_2_tmp = attn_score.clone() / attn_score.clone().sum();
    println!("Attention weights: {:?}", attn_weights_2_tmp);
    println!("Sum {:?}:", attn_weights_2_tmp.sum());

    let attn_weights_2_naive = softmax_naive(attn_score.clone());
    println!("Attention weights: {:?}", attn_weights_2_naive);
    println!("Sum {:?}:", attn_weights_2_naive.sum());

    let attn_weights_2 = activation::softmax(attn_score.clone(), 0);
    println!("Attention weights: {:?}", attn_weights_2);
    println!("Sum {:?}:", attn_weights_2.clone().sum());

    let mut context_vec_2 = Tensor::<B, 1>::zeros([d1], &device);
    for i in 0..d0 {
        let w = attn_weights_2.clone().select(0, [i].into());
        let x = inputs.clone().select(0, [i].into()).reshape([d1]);

        context_vec_2 = context_vec_2 + w * x;
    }
    println!("\n{context_vec_2:?}");

    // 3.3.2. Computing attention weights for all input tokens
    let mut attn_scores = Tensor::<B, 2>::zeros([d0, d0], &device);
    for (i, x) in inputs.clone().iter_dim(0).enumerate() {
        for (j, y) in inputs.clone().iter_dim(0).enumerate() {
            let z = x.clone().mul(y).sum().unsqueeze();

            attn_scores = attn_scores.slice_assign([i..(i + 1), j..(j + 1)], z);
        }
    }
    println!("{attn_scores:?}\n");

    let attn_scores = inputs.clone().matmul(inputs.clone().transpose());
    println!("{attn_scores:?}\n");

    let dim = attn_scores.dims().len() - 1;
    let attn_weights = activation::softmax(attn_scores.clone(), dim);
    println!("{attn_weights:?}\n");

    let row2_sum = attn_weights.clone().select(0, [1].into()).sum();
    print!("{row2_sum:?}");
    println!(
        "All row sums:\n{:?}",
        attn_weights.clone().sum_dim(attn_weights.clone().dims().len() - 1)
    );

    let all_context_vec = attn_weights.clone().matmul(inputs.clone());
    println!("{all_context_vec:?}");
}

fn softmax_naive(x: Tensor<B, 1>) -> Tensor<B, 1> {
    x.clone().exp() / x.clone().exp().sum()
}
