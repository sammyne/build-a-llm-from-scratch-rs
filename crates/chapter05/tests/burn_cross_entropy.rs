use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::loss;

type B = NdArray<f32>;

#[test]
fn cross_entropy_with_logits() {
    let device = <B as Backend>::Device::default();

    let logits = Tensor::<B, 2>::from_floats(
        [
            [1.0, 2.0, 3.0], // 样本1
            [0.5, 1.5, 0.1], // 样本2
        ],
        &device,
    );

    let target_probs = Tensor::<B, 2>::from_floats(
        [
            [0.2, 0.3, 0.5], // 样本1的目标分布
            [0.8, 0.1, 0.1], // 样本2的目标分布
        ],
        &device,
    );

    let loss = loss::cross_entropy_with_logits(logits, target_probs);
    let got = format!("{:.4}", loss.into_scalar());
    const EXPECT: &str = "1.2633";
    assert_eq!(EXPECT, got, "mismatched loss against PyTorch");
}
