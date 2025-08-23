use anyhow::Context;
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{Autodiff, LibTorch};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use chapter05::config::GPT_124M;

type B = Autodiff<LibTorch>;
// type B = Autodiff<Cuda>;

fn main() -> anyhow::Result<()> {
    let device = &LibTorchDevice::Cpu;

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let _model = GPT_124M
        .init::<B>(device)
        .load_file("gpt_124m_trained.burn", &recorder, device)
        .context("load model")?;

    Ok(())
}
