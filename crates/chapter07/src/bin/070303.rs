use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::Tensor;
use chapter07::PAD_TOKEN_ID;

type B = NdArray;

type Device = <B as Backend>::Device;

fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let inputs_1 = [0, 1, 2, 3, 4].as_slice();
    let inputs_2 = [5, 6].as_slice();
    let inputs_3 = [7, 8, 9].as_slice();
    let batch = [inputs_1, inputs_2, inputs_3];

    let (inputs, targets) = custom_collate_draft_2::<B, _>(&batch, None, device);
    println!("\n{inputs}");
    println!("{targets}");

    Ok(())
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
