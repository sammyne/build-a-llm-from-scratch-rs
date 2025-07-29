use std::path::Path;

use anyhow::Context as _;
use burn::backend::{Autodiff, LibTorch};
use burn::module::Module;
use burn::nn::LinearConfig;
use burn::prelude::Backend;
use burn::tensor::{Tensor, s};
use chapter04::GptModel;
use chapter05::gpt2;
use chapter05::utils::Tokenizer as _;
use chapter06::dataset::{self, DataLoaderOptions, LoadCsvOptions, SpamDataset};
use chapter06::loss;
use chapter06::utils::RequireGradMapper;
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let data_dir = &Path::new("gpt2/gpt2/124M");
    let (settings, params) = {
        let (mut s, p) = gpt2::load_settings_and_params(&data_dir).expect("load gpt2 config");
        s.drop_rate = 0.0;
        (s, p)
    };

    let mut model = GptModel::<B>::new(&settings, device);

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
    println!(
        "Last output token: {}",
        outputs.clone().slice(s![.., -1, ..]).squeeze::<2>(1)
    );

    let dim = outputs.dims().len() - 1;
    // 使用 softmax
    // let probas = activation::softmax(outputs.slice(s![.., -1, ..]), dim);
    // let label = probas.clone().argmax(dim);
    // println!("Class label: {}", label.into_scalar());

    // 不是用 softmax
    let logits = outputs.slice(s![.., -1, ..]);
    let label = logits.argmax(dim);
    println!("Class label: {}", label.into_scalar());

    B::seed(123);

    let opts = LoadCsvOptions::new("train.csv", &tokenizer, device);
    let train_dataset = SpamDataset::<B>::load_csv(opts).context("load train dataset")?;
    // 由于随机种子和算法不一致，因此和书上的有差异。
    println!("max-length(train-dataset): {}", train_dataset.max_length);

    let opts = LoadCsvOptions::new("validation.csv", &tokenizer, device);
    let validation_dataset = SpamDataset::<B>::load_csv(opts).context("load validation dataset")?;

    let opts = LoadCsvOptions::new("test.csv", &tokenizer, device);
    let test_dataset = SpamDataset::<B>::load_csv(opts).context("load test dataset")?;

    // 对于训练集，丢弃不完整的最后一批。
    let train_loader = dataset::load(train_dataset, DataLoaderOptions::default().with_drop_last(true));
    let validation_loader = dataset::load(validation_dataset, Default::default());
    let test_loader = dataset::load(test_dataset, Default::default());

    let train_accuracy = loss::calc_accuracy_loader(train_loader.as_ref(), &model, device, Some(10));
    let val_accuracy = loss::calc_accuracy_loader(validation_loader.as_ref(), &model, device, Some(10));
    let test_accuracy = loss::calc_accuracy_loader(test_loader.as_ref(), &model, device, Some(10));
    println!("Training accuracy: {:.2}", train_accuracy * 100.0);
    println!("Validation accuracy: {:.2}", val_accuracy * 100.0);
    println!("Test accuracy: {:.2}", test_accuracy * 100.0);

    let train_loss = loss::calc_loss_loader(train_loader.as_ref(), model.clone(), device, None);
    let val_loss = loss::calc_loss_loader(validation_loader.as_ref(), model.clone(), device, None);
    let test_loss = loss::calc_loss_loader(test_loader.as_ref(), model.clone(), device, None);
    println!("Training loss: {:.3}", train_loss);
    println!("Validation loss: {:.3}", val_loss);
    println!("Test loss: {:.3}", test_loss);

    Ok(())
}
