use burn::backend::NdArray;
use burn::prelude::*;

type B = NdArray;

type Device = <B as Backend>::Device;

fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let inputs_1 = [0, 1, 2, 3, 4].as_slice();
    let inputs_2 = [5, 6].as_slice();
    let inputs_3 = [7, 8, 9].as_slice();
    let batch = [inputs_1, inputs_2, inputs_3];

    let (inputs, targets) = chapter07::utils::custom_collate_fn::<B, _>(&batch, None, None, None, device);
    println!("{inputs}");
    println!("{targets}");

    Ok(())
}
