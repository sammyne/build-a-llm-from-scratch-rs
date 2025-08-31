use anyhow::Context as _;
use burn::backend::{Autodiff, LibTorch};
use burn::module::Module;
use burn::nn::LinearConfig;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use chapter05::utils::Tokenizer as _;
use chapter06::utils::{self, RequireGradMapper};
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let model = utils::load_gpt2("gpt2/124M", device).context("load model")?;

    B::seed(123);

    let mut model = model.no_grad();

    const NUM_CLASSES: usize = 2;
    let emb_dim = model.tok_emb.weight.dims()[1];
    model.out_head = LinearConfig::new(emb_dim, NUM_CLASSES).with_bias(true).init(device);

    let trf_block = model.trf_blocks.last_mut().context("miss last transfomer block")?;
    *trf_block = trf_block.clone().map(&mut RequireGradMapper);

    model.final_norm = model.final_norm.clone().map(&mut RequireGradMapper);

    let tokenizer = Encoding::gpt2();

    let inputs: Tensor<B, 2, _> = tokenizer.tokenize("Do you have time").to_device(device);

    let outputs = model.no_grad().forward(inputs);
    println!("Outputs:\n{outputs}");
    println!("Outputs dimensions:\n{:?}", outputs.shape());

    Ok(())
}
