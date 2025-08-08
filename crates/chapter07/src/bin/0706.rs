use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use burn::LearningRate;
use burn::backend::{Autodiff, LibTorch};
use burn::data::dataloader::DataLoader;
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use chapter04::GptModel;
use chapter05::gpt2;
use chapter05::utils::Tokenizer;
use chapter07::dataset::Batch;
use chapter07::{loss, utils};
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let data_dir = &Path::new("gpt2/gpt2/355M");
    let (settings, params) = {
        let (mut s, p) = gpt2::load_settings_and_params(&data_dir).expect("load gpt2 config");
        s.drop_rate = 0.0;
        (s, p)
    };

    let mut model = GptModel::<B>::new(&settings, device);

    gpt2::load_weights_into_gpt2(params, &mut model).context("load weights into model")?;

    B::seed(123);
    let tokenizer = Encoding::gpt2();

    let start_ctx = {
        let v = utils::load_and_split_data("instruction-data.json")
            .context("load and split data")?
            .2;
        utils::format_input(&v[0])
    };

    let (train_loader, _test_loader, val_loader) =
        chapter07::dataset::load_and_split("instruction-data.json", &tokenizer)
            .context("load and split data loader")?;

    let train_loss =
        chapter07::loss::calc_loss_loader(train_loader.as_ref(), &model.clone().no_grad(), 5.into(), device);
    let val_loss = chapter07::loss::calc_loss_loader(val_loader.as_ref(), &model.clone().no_grad(), 5.into(), device);

    println!("Training loss: {}", train_loss);
    println!("Validation loss: {}", val_loss);

    let start = Instant::now();
    B::seed(123);

    // TODO：查明在哪里关闭了梯度传递
    let model = model.map(&mut chapter06::utils::RequireGradMapper);

    let optimizer = AdamWConfig::new().with_weight_decay(0.1).init::<B, GptModel<B>>();
    let opts = TrainOpts {
        model,
        train_loader: train_loader.as_ref(),
        val_loader: val_loader.as_ref(),
        optimizer,
        device: &device,
        epoches: 2,
        eval_freq: 5,
        eval_iter: 5,
        start_context: &start_ctx,
        tokenizer: &tokenizer,
        lr: 0.00005,
    };

    let (model, ..) = train_model_simple(opts);

    println!("Training completed in {:?}", start.elapsed());

    // 保存模型用于后续教程
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    const MODEL_PATH: &str = "gpt_355m_model";
    model.save_file(MODEL_PATH, &recorder).expect("save model");

    Ok(())
}

struct TrainOpts<'a, B, O, T>
where
    B: AutodiffBackend,
    O: Optimizer<GptModel<B>, B>,
    T: Tokenizer<B::InnerBackend>,
{
    model: GptModel<B>,
    train_loader: &'a dyn DataLoader<B, Batch<B>>,
    val_loader: &'a dyn DataLoader<B, Batch<B>>,
    optimizer: O,
    device: &'a B::Device,
    epoches: usize,
    eval_freq: usize,
    eval_iter: usize,
    start_context: &'a str,
    tokenizer: &'a T,
    lr: LearningRate,
}

struct EvaluateOpts<'a, B: AutodiffBackend> {
    model: GptModel<B>,
    train_loader: &'a dyn DataLoader<B, Batch<B>>,
    val_loader: &'a dyn DataLoader<B, Batch<B>>,
    device: &'a B::Device,
    eval_iter: usize,
}

fn evaluate_model<B>(opts: EvaluateOpts<'_, B>) -> (f32, f32)
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let model = opts.model.no_grad();
    let train_loss = loss::calc_loss_loader(opts.train_loader, &model, opts.eval_iter.into(), opts.device);
    let val_loss = loss::calc_loss_loader(opts.val_loader, &model, opts.eval_iter.into(), opts.device);

    (train_loss, val_loss)
}

fn generate_and_print_sample<B, T>(model: GptModel<B>, tokenizer: &T, device: &B::Device, start_context: &str)
where
    B: AutodiffBackend,
    T: Tokenizer<B::InnerBackend>,
{
    let model = model.valid();

    let context_size = model.pos_emb.weight.shape().dims[0];
    let encoded = tokenizer.tokenize(start_context).to_device(device);
    let token_ids = chapter04::utils::generate_text_simple(&model, encoded, 50, context_size);
    let decoded_text = tokenizer.detokenize(token_ids).expect("decode text");
    println!("{}", decoded_text.replace('\n', " "));
}

fn train_model_simple<B, O, T>(opts: TrainOpts<'_, B, O, T>) -> (GptModel<B>, O, Vec<f32>, Vec<f32>, Vec<usize>)
where
    B: AutodiffBackend<FloatElem = f32>,
    O: Optimizer<GptModel<B>, B>,
    T: Tokenizer<B::InnerBackend>,
{
    let TrainOpts {
        mut model,
        train_loader,
        val_loader,
        mut optimizer,
        device,
        epoches,
        eval_freq,
        eval_iter,
        start_context,
        tokenizer,
        lr,
    } = opts;

    let mut train_losses = vec![];
    let mut val_losses = vec![];
    let mut track_tokens_seen = vec![];

    let mut token_seen = 0;
    let eval_freq = eval_freq as i32;
    let mut global_step = -1i32;

    for epoch in 1..=epoches {
        for (input_batch, target_batch) in train_loader.iter() {
            let loss = loss::calc_loss_batch(input_batch.clone(), target_batch, &model, device);

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            // TODO: 更新学习率
            model = optimizer.step(lr, model, grads);

            token_seen += input_batch.shape().num_elements();
            global_step += 1;

            if global_step % eval_freq != 0 {
                continue;
            }

            // 评估模型
            let opts = EvaluateOpts {
                model: model.clone(),
                train_loader,
                val_loader,
                device,
                eval_iter,
            };
            let (train_loss, val_loss) = evaluate_model(opts);
            train_losses.push(train_loss);
            val_losses.push(val_loss);
            track_tokens_seen.push(token_seen);
            println!("Ep {epoch} (Step: {global_step:06}): Train Loss: {train_loss:.3}, Val Loss: {val_loss:.3}");
        }

        generate_and_print_sample(model.clone(), tokenizer, device, start_context);
    }

    (model, optimizer, train_losses, val_losses, track_tokens_seen)
}
