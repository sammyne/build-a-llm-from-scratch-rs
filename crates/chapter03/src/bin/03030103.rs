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

    let query = inputs.clone().select(0, [1].into()).flatten(0, 1);
    let [d0, d1] = inputs.shape().dims();

    let mut attn_score_2 = Tensor::<B, 1>::empty([d0], &device);
    for (i, x) in inputs.clone().iter_dim(0).enumerate() {
        let x = x.reshape([d1]);
        attn_score_2 = attn_score_2.slice_assign([i..(i + 1)], x.mul(query.clone()).sum());
    }

    let attn_weights_2_naive = softmax_naive(attn_score_2.clone());
    println!("Attention weights: {}", attn_weights_2_naive);
    println!("Sum {}:", attn_weights_2_naive.sum());
}

fn softmax_naive(x: Tensor<B, 1>) -> Tensor<B, 1> {
    x.clone().exp() / x.exp().sum()
}
