use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context as _;
use burn::LearningRate;
use burn::backend::{Autodiff, LibTorch};
use burn::data::dataloader::DataLoader;
use burn::module::Module;
use burn::nn::LinearConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder as _};
use burn::tensor::backend::AutodiffBackend;
use chapter04::GptModel;
use chapter05::gpt2;
use chapter06::dataset::{self, Batch, DataLoaderOptions, LoadCsvOptions, SpamDataset};
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

    let opts = LoadCsvOptions::new("train.csv", &tokenizer, device);
    let train_dataset = SpamDataset::<B>::load_csv(opts).context("load train dataset")?;

    let opts = LoadCsvOptions::new("validation.csv", &tokenizer, device);
    let validation_dataset = SpamDataset::<B>::load_csv(opts).context("load validation dataset")?;

    let opts = LoadCsvOptions::new("test.csv", &tokenizer, device);
    let test_dataset = SpamDataset::<B>::load_csv(opts).context("load test dataset")?;

    // 对于训练集，丢弃不完整的最后一批。
    let train_loader = dataset::load(train_dataset, DataLoaderOptions::default().with_drop_last(true));
    let validation_loader = dataset::load(validation_dataset, Default::default());
    let test_loader = dataset::load(test_dataset, Default::default());

    let start = Instant::now();
    B::seed(123);

    let optimizer = AdamWConfig::new().with_weight_decay(0.1).init::<B, GptModel<B>>();

    let opts = TrainOpts {
        model: model.clone(),
        train_loader: train_loader.as_ref(),
        val_loader: validation_loader.as_ref(),
        optimizer,
        device: &device,
        epoches: 5,
        eval_freq: 50,
        eval_iter: 5,
        lr: 5e-5,
    };

    let TrainOutput { model, .. } = train_classifer_simple(opts);

    let execution_time_minutes = start.elapsed().as_secs_f64() / 60.0;
    println!("Training completed in {execution_time_minutes:.2} minutes");

    let train_accuracy = loss::calc_accuracy_loader(train_loader.as_ref(), &model, device, None);
    let val_accuracy = loss::calc_accuracy_loader(validation_loader.as_ref(), &model, device, None);
    let test_accuracy = loss::calc_accuracy_loader(test_loader.as_ref(), &model, device, None);

    println!("Train accuracy: {:.2}%", train_accuracy * 100.0);
    println!("Validation accuracy: {:.2}%", val_accuracy * 100.0);
    println!("Test accuracy: {:.2}%", test_accuracy * 100.0);

    // 保存模型用于后续教程
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    const MODEL_PATH: &str = "spam-classifier-model";
    model.save_file(MODEL_PATH, &recorder).expect("save model");

    // let optimizer_path = PathBuf::from("spam-classifier-optimizer");
    // recorder
    //     .record(optimizer.to_record(), optimizer_path)
    //     .context("save optimizer")?;

    Ok(())
}

struct EvaluateOpts<'a, B: AutodiffBackend> {
    model: GptModel<B>,
    train_loader: &'a dyn DataLoader<B, Batch<B>>,
    val_loader: &'a dyn DataLoader<B, Batch<B>>,
    device: &'a B::Device,
    eval_iter: usize,
}

struct TrainOpts<'a, B, O>
where
    B: AutodiffBackend,
    O: Optimizer<GptModel<B>, B>,
{
    model: GptModel<B>,
    train_loader: &'a dyn DataLoader<B, Batch<B>>,
    val_loader: &'a dyn DataLoader<B, Batch<B>>,
    optimizer: O,
    device: &'a B::Device,
    epoches: usize,
    eval_freq: usize,
    eval_iter: usize,
    lr: LearningRate,
}

struct TrainOutput<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<GptModel<B>, B>,
{
    model: GptModel<B>,
    optimizer: O,
    train_losses: Vec<f32>,
    val_losses: Vec<f32>,
    train_accs: Vec<f32>,
    val_accs: Vec<f32>,
    examples_seen: usize,
}

fn evaluate_model<B>(opts: EvaluateOpts<'_, B>) -> (f32, f32)
where
    B: AutodiffBackend<FloatElem = f32>,
{
    let model = opts.model.no_grad();

    let num_batches = opts.eval_iter.into();
    let train_loss = loss::calc_loss_loader(opts.train_loader, model.clone(), opts.device, num_batches);
    let val_loss = loss::calc_loss_loader(opts.val_loader, model, opts.device, num_batches);

    (train_loss, val_loss)
}

fn train_classifer_simple<B, O>(opts: TrainOpts<'_, B, O>) -> TrainOutput<B, O>
where
    B: AutodiffBackend<FloatElem = f32>,
    O: Optimizer<GptModel<B>, B>,
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
        lr,
    } = opts;

    let mut train_losses = vec![];
    let mut val_losses = vec![];
    let mut train_accs = vec![];
    let mut val_accs = vec![];

    let mut examples_seen = 0;
    let eval_freq = eval_freq as i32;
    let mut global_step = -1i32;

    for epoch in 1..=epoches {
        for (input_batch, target_batch) in train_loader.iter() {
            let loss = loss::calc_loss_batch(input_batch.clone(), target_batch, &model, device);

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            // TODO: 更新学习率
            model = optimizer.step(lr, model, grads);

            examples_seen += input_batch.dims()[0];
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

            println!("Ep {epoch} (Step: {global_step:06}): Train Loss: {train_loss:.3}, Val Loss: {val_loss:.3}");
        }

        let train_accuracy = loss::calc_accuracy_loader(train_loader, &model, device, Some(eval_iter));
        let val_accuracy = loss::calc_accuracy_loader(val_loader, &model, device, Some(eval_iter));
        println!(
            "Training accuracy: {:.2}% | Validation accuracy: {:.2}%",
            train_accuracy * 100.0,
            val_accuracy * 100.0
        );
        train_accs.push(train_accuracy);
        val_accs.push(val_accuracy);
    }

    let out = TrainOutput {
        model,
        optimizer,
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        examples_seen,
    };

    out
}
