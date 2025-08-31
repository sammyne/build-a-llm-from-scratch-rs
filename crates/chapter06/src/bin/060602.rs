use std::path::Path;

use anyhow::Context as _;
use burn::backend::{Autodiff, LibTorch};
use burn::module::Module;
use burn::nn::LinearConfig;
use burn::prelude::Backend;
use burn::tensor::{Tensor, activation, s};
use chapter04::GPT_124M;
use chapter05::gpt2;
use chapter05::utils::Tokenizer as _;
use chapter06::utils::RequireGradMapper;
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let data_dir = &Path::new("gpt2/124M");
    let (settings, params) = {
        let (mut s, p) = gpt2::load_settings_and_params(&data_dir).expect("load gpt2 config");
        s.drop_rate = 0.0;
        (s, p)
    };

    let mut model = GPT_124M
        .with_context_length(settings.context_length)
        .with_qkv_bias(true)
        .init::<B>(device);

    gpt2::load_weights_into_gpt2(params, &mut model).context("load weights into model")?;

    B::seed(123);

    let mut model = model.no_grad();

    const NUM_CLASSES: usize = 2;
    model.out_head = LinearConfig::new(settings.emb_dim, NUM_CLASSES)
        .with_bias(true)
        .init(device);

    let trf_block = model.trf_blocks.last_mut().context("miss last transfomer block")?;
    *trf_block = trf_block.clone().map(&mut RequireGradMapper);

    model.final_norm = model.final_norm.clone().map(&mut RequireGradMapper);

    let tokenizer = Encoding::gpt2();

    let inputs: Tensor<B, 2, _> = tokenizer.tokenize("Do you have time").to_device(device);

    let outputs = model.clone().no_grad().forward(inputs);

    let dim = outputs.dims().len() - 1;
    // 使用 softmax
    let probas = activation::softmax(outputs.slice(s![.., -1, ..]), dim);
    let label = probas.clone().argmax(dim);
    println!("Class label: {}", label.into_scalar());

    Ok(())
}
