use burn::backend::NdArray;
use burn::tensor::Tensor;

type B = NdArray<f32>;

fn main() {
    let device = <B as burn::prelude::Backend>::Device::default();

    let a = Tensor::<B, 4, _>::from_floats(
        [[
            [
                [0.2745, 0.6584, 0.2775, 0.8573],
                [0.8993, 0.0390, 0.9268, 0.7388],
                [0.7179, 0.7058, 0.9156, 0.4340],
            ],
            [
                [0.0772, 0.3565, 0.1479, 0.5331],
                [0.4066, 0.2318, 0.4545, 0.9737],
                [0.4606, 0.5159, 0.4220, 0.5786],
            ],
        ]],
        &device,
    );

    let b = a.clone().matmul(a.clone().transpose());
    println!("{b:?}");

    let first_head = a.clone().slice([0..1, 0..1]).squeeze::<3>(0).squeeze::<2>(0);
    let first_res = first_head.clone().matmul(first_head.transpose());
    println!("\n{first_res:?}\n");

    let second_head = a.clone().slice([0..1, 1..2]).squeeze::<3>(0).squeeze::<2>(0);
    let second_res = second_head.clone().matmul(second_head.transpose());
    println!("\n{second_res:?}\n");
}
