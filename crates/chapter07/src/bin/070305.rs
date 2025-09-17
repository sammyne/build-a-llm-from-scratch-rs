use burn::backend::NdArray;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::tensor::Tensor;

type B = NdArray;

type Device = <B as Backend>::Device;

fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let logits_1 = Tensor::<B, 2>::from_floats([[-1.0, 1.0], [-0.5, 1.5]], device);
    let targets_1 = Tensor::<B, 1, Int>::from_ints([0, 1], device);
    let loss_1 = CrossEntropyLossConfig::new()
        .init::<B>(device)
        .forward(logits_1, targets_1);
    println!("Loss 1: {loss_1}");

    let logits_2 = Tensor::<B, 2>::from_floats([[-1.0, 1.0], [-0.5, 1.5], [-0.5, 1.5]], device);
    let targets_2 = Tensor::<B, 1, Int>::from_ints([0, 1, 1], device);
    let loss_2 = CrossEntropyLossConfig::new()
        .init::<B>(device)
        .forward(logits_2.clone(), targets_2);
    println!("\nLoss 2: {loss_2}");

    let targets_3 = Tensor::<B, 1, Int>::from_ints([0, 1, -100], device);
    let loss_3 = chapter07::loss::CrossEntropyLossConfig::new()
        .init()
        .forward(logits_2, targets_3);
    println!("\nLoss 3: {loss_3}");
    println!("loss_1 == loss_3: {}", loss_1.equal(loss_3).into_scalar());

    Ok(())
}
