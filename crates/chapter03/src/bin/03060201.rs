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
    println!("{b}");
}
