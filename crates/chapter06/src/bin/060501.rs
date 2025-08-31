use burn::backend::{Autodiff, LibTorch};
use burn::tensor::Tensor;
use chapter05::utils::Tokenizer as _;
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let tokenizer = Encoding::gpt2();

    let inputs: Tensor<B, 2, _> = tokenizer.tokenize("Do you have time");
    // let inputs = inputs.unsqueeze::<1>();
    println!("Inputs: {inputs}");
    println!("Inputs dimensions: {:?}", inputs.shape());

    Ok(())
}
