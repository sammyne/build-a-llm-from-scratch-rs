use burn::prelude::Backend;
use burn::tensor::{Distribution, Tensor, activation};
use chapter03::attention::v1::SelfAttentionV1;

type B = burn::backend::ndarray::NdArray<f32>;

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

    let x2 = inputs.clone().select(0, [1].into());
    let d_in = inputs.dims()[1];
    let d_out = 2;

    let distribution = Distribution::Uniform(0.0, 1.0);

    let q = Tensor::<B, 2_>::random([d_in, d_out], distribution, &device);
    let k = Tensor::<B, 2_>::random([d_in, d_out], distribution, &device);
    let v = Tensor::<B, 2_>::random([d_in, d_out], distribution, &device);

    let q2 = x2.clone().matmul(q.clone());
    // let _k2 = x2.clone().matmul(k);
    // let _v2 = x2.clone().matmul(v);
    println!("q2\n{q2:?}");

    let keys = inputs.clone().matmul(k.clone());
    let values = inputs.clone().matmul(v.clone());
    println!("keys.shape = {:?}", keys.shape());
    println!("values.shape = {:?}", values.shape());

    let k2 = keys.clone().select(0, [1].into());
    let attn_score22 = q2.clone().mul(k2).sum();
    println!("attn_score22\n{attn_score22:?}");

    let attn_scores2 = q2.clone().matmul(keys.clone().transpose());
    println!("attn_scores2\n{attn_scores2:?}");

    let d_k = *keys.dims().last().expect("get -1-th dim for keys") as f32;
    let dim = attn_scores2.dims().len() - 1;
    let attn_weights2 = activation::softmax(attn_scores2.clone() / d_k.sqrt(), dim);
    println!("attn_weights2\n{attn_weights2:?}");

    let context_vec2 = attn_weights2.clone().matmul(values.clone());
    println!("context_vec2\n{context_vec2:?}");

    let sa_v1 = SelfAttentionV1::<B>::new(d_in, d_out);
    let context_vec3 = sa_v1.forward(inputs.clone());
    println!("context_vec3\n{context_vec3:?}");

    let sa_v2 = SelfAttentionV2::<B>::new(d_in, d_out, false);
    let context_vec4 = sa_v2.forward(inputs.clone());
    println!("context_vec4\n{context_vec4:?}");
}
