use burn::backend::NdArray;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::prelude::*;
use burn::tensor::Tensor;
use chapter07::PAD_TOKEN_ID;
use tiktoken::ext::Encoding;

type B = NdArray;

type Device = <B as Backend>::Device;

fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let tokenizer = Encoding::gpt2();

    let allowed_special = ["<|endoftext|>"].into();
    let got = tokenizer.encode("<|endoftext|>", &allowed_special);
    assert_eq!(PAD_TOKEN_ID, got[0], "unexpected token id for <|endoftext|>");

    let inputs_1 = [0, 1, 2, 3, 4].as_slice();
    let inputs_2 = [5, 6].as_slice();
    let inputs_3 = [7, 8, 9].as_slice();
    let batch = [inputs_1, inputs_2, inputs_3];

    println!("\n{}", custom_collate_draft_1::<B, _>(&batch, None, device));

    let (inputs, targets) = custom_collate_draft_2::<B, _>(&batch, None, device);
    println!("\n{inputs}");
    println!("{targets}");

    let (inputs, targets) = chapter07::utils::custom_collate_fn::<B, _>(&batch, None, None, None, device);
    println!("\n{inputs}");
    println!("{targets}");

    let logits_1 = Tensor::<B, 2>::from_floats([[-1.0, 1.0], [-0.5, 1.5]], device);
    let targets_1 = Tensor::<B, 1, Int>::from_ints([0, 1], device);
    let loss_1 = CrossEntropyLossConfig::new()
        .init::<B>(device)
        .forward(logits_1, targets_1);
    println!("\nLoss 1: {loss_1}");

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

fn custom_collate_draft_1<B: Backend, T: AsRef<[u32]>>(
    batch: &[T],
    pad_token_id: Option<u32>,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch_max_length = batch.iter().map(|x| x.as_ref().len()).max().unwrap_or(0) + 1;
    let pad_token_id = pad_token_id.unwrap_or(PAD_TOKEN_ID);

    let mut inputs_lst = Vec::with_capacity(batch.len());
    for item in batch {
        let mut padded = item.as_ref().to_vec();

        padded.resize(batch_max_length, pad_token_id);
        let inputs = Tensor::<B, 1, Int>::from_ints(&padded[..batch_max_length - 1], device);
        inputs_lst.push(inputs);
    }

    Tensor::stack(inputs_lst, 0)
}

fn custom_collate_draft_2<B: Backend, T: AsRef<[u32]>>(
    batch: &[T],
    pad_token_id: Option<u32>,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let batch_max_length = batch.iter().map(|x| x.as_ref().len()).max().unwrap_or(0) + 1;
    let pad_token_id = pad_token_id.unwrap_or(PAD_TOKEN_ID);

    let mut inputs_lst = Vec::with_capacity(batch.len());
    let mut targets_lst = Vec::with_capacity(batch.len());
    for item in batch {
        let mut padded = item.as_ref().to_vec();

        padded.resize(batch_max_length, pad_token_id);
        let inputs = Tensor::<B, 1, Int>::from_ints(&padded[..batch_max_length - 1], device);
        inputs_lst.push(inputs);

        let targets = Tensor::<B, 1, Int>::from_ints(&padded[1..], device);
        targets_lst.push(targets);
    }

    (Tensor::stack(inputs_lst, 0), Tensor::stack(targets_lst, 0))
}
