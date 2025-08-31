use std::fs::OpenOptions;
use std::usize;

use anyhow::Context;
use burn::LearningRate;
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{Autodiff, LibTorch};
use burn::data::dataloader::DataLoader;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use chapter02::dataset::{self, Batch, LoaderV1Options};
use chapter02::verdict;
use chapter04::GptModel;
use chapter05::config::GPT_124M;
use chapter05::loss;
use chapter05::utils::Tokenizer;
use serde::{Deserialize, Serialize};
use tiktoken::ext::Encoding;

type B = Autodiff<LibTorch>;
// type B = Autodiff<Cuda>;

fn main() -> anyhow::Result<()> {
    let tokenizer = Encoding::gpt2();

    let text_data = verdict::load().context("load verdict")?;

    const TRAIN_RATIO: f64 = 0.9;
    let (train_data, val_data) = {
        let i = (text_data.len() as f64 * TRAIN_RATIO) as usize;
        text_data.split_at(i)
    };

    B::seed(123);
    let device = if !minikit::cuda::is_available() {
        &LibTorchDevice::Cpu
    } else {
        println!("using CUDA");
        &LibTorchDevice::Cuda(0)
    };

    let train_loader = {
        let opts = LoaderV1Options {
            batch_size: 2,
            max_length: GPT_124M.context_length,
            stride: GPT_124M.context_length,
            drop_last: true,
            shuffle_seed: Some(123),
            ..Default::default()
        };
        dataset::create_dataloader_v1::<B, _>(train_data, &tokenizer, opts).context("load train data")?
    };

    let val_loader = {
        let opts = LoaderV1Options {
            batch_size: 2,
            max_length: GPT_124M.context_length,
            stride: GPT_124M.context_length,
            drop_last: false,
            ..Default::default()
        };
        dataset::create_dataloader_v1::<B, _>(val_data, &tokenizer, opts).context("load validation data")?
    };

    // 警告：Module::to_device 转移后的模型不再支持反向传播。
    // 详情参见 burn 的官方文档：https://docs.rs/burn/0.17.1/burn/module/trait.Module.html#tymethod.to_device
    let model = GPT_124M.init(device);

    let optimizer = AdamWConfig::new().with_weight_decay(0.1).init::<B, GptModel<B>>();

    let opts = TrainOpts {
        model,
        train_loader: train_loader.as_ref(),
        val_loader: val_loader.as_ref(),
        optimizer,
        device: &device,
        epoches: 10,
        eval_freq: 5,
        eval_iter: 5,
        start_context: "Every effort moves you",
        tokenizer: &tokenizer,
        lr: 0.0004,
    };

    let overview = train_model_simple(opts);
    overview.save("train-overview.json")?;

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

#[derive(Serialize, Deserialize)]
struct TrainOverview {
    epoches: usize,
    train_losses: Vec<f32>,
    val_losses: Vec<f32>,
    track_tokens_seen: Vec<usize>,
}

struct EvaluateOpts<'a, B: AutodiffBackend> {
    model: GptModel<B>,
    train_loader: &'a dyn DataLoader<B, Batch<B>>,
    val_loader: &'a dyn DataLoader<B, Batch<B>>,
    device: &'a B::Device,
    eval_iter: usize,
}

impl TrainOverview {
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .context("open file")?;
        serde_json::to_writer(&mut f, self).context("json dumps")
    }
}

fn evaluate_model<B>(opts: EvaluateOpts<'_, B>) -> (f32, f32)
where
    B: AutodiffBackend<FloatElem = f32>,
{
    // TODO: 评估没有显式调用 valid 函数排除 dropout 的影响有多大。
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

fn train_model_simple<B, O, T>(opts: TrainOpts<'_, B, O, T>) -> TrainOverview
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

    // 保存模型用于后续教程
    const MODEL_PATH: &str = "gpt_124m_trained.burn";
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.save_file(MODEL_PATH, &recorder).expect("save model");

    let out = TrainOverview {
        epoches,
        train_losses,
        val_losses,
        track_tokens_seen,
    };

    out
}
