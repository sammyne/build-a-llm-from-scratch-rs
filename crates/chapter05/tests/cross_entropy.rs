use burn::backend::NdArray;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use chapter05::utils;

type B = NdArray<f32>;

#[test]
fn cross_entropy_2d() {
    let device = <B as Backend>::Device::default();

    struct Case {
        logits: Tensor<B, 2>,
        target_indices: Tensor<B, 1, Int>,
        expect: &'static str,
    }

    let test_vector = vec![
        Case {
            logits: Tensor::<B, 2>::from_floats(
                [
                    [1.0, 2.0, 3.0], // 样本 1
                    [1.0, 2.0, 3.0], // 样本 2
                ],
                &device,
            ),
            target_indices: Tensor::<B, 1, Int>::from_ints([2, 0], &device),
            expect: "1.4076",
        },
        Case {
            logits: Tensor::<B, 2>::from_floats(
                [
                    [2.0, 1.0, 0.1, 0.5, 0.3], // 样本1
                    [0.5, 1.5, 2.5, 0.1, 0.2], // 样本2
                    [1.0, 2.0, 0.5, 0.1, 3.0], // 样本3
                ],
                &device,
            ),
            target_indices: Tensor::<B, 1, Int>::from_ints([0, 2, 4], &device),
            expect: "0.5587",
        },
    ];

    for (i, c) in test_vector.into_iter().enumerate() {
        let got = utils::cross_entropy(c.logits, c.target_indices).into_scalar();
        let got = format!("{got:.4}");
        assert_eq!(c.expect, got, "#{i} mismatched against PyTorch");
    }
}

#[test]
fn cross_entropy_3d() {
    let device = <B as Backend>::Device::default();

    struct Case {
        logits: Tensor<B, 3>,
        target_indices: Tensor<B, 2, Int>,
        expect: &'static str,
    }

    let test_vector = vec![Case {
        logits: Tensor::<B, 3>::from_floats(
            [
                // 样本1
                [
                    [1.0, 2.0, 0.5], // 位置1
                    [0.5, 1.5, 0.1], // 位置2
                    [2.0, 0.5, 0.1], // 位置3
                ],
                // 样本2
                [
                    [0.1, 0.2, 3.0], // 位置1
                    [1.0, 2.0, 0.5], // 位置2
                    [0.5, 1.5, 0.1], // 位置3
                ],
            ],
            &device,
        ),
        target_indices: Tensor::<B, 2, Int>::from_ints(
            [
                [1, 0, 2], // 样本1: 位置1=类别1, 位置2=类别0, 位置3=类别2
                [2, 1, 0], // 样本2: 位置1=类别2, 位置2=类别1, 位置3=类别0
            ],
            &device,
        ),
        expect: "1.0355",
    }];

    for (i, c) in test_vector.into_iter().enumerate() {
        let got = utils::cross_entropy(c.logits, c.target_indices).into_scalar();
        let got = format!("{got:.4}");
        assert_eq!(c.expect, got, "#{i} mismatched against PyTorch");
    }
}
