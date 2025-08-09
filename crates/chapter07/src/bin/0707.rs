use std::fs::File;
use std::path::Path;

use anyhow::Context as _;
use burn::backend::LibTorch;
use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use chapter04::GptModel;
use chapter05::gpt2;
use chapter05::utils::Tokenizer;
use chapter07::utils::{self, DataWithModelResponse};
use indicatif::ProgressBar;
use tiktoken::ext::Encoding;

type B = LibTorch;

type Device = <B as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let model_path = Path::new("gpt-355m-model-sft.mpk");
    if !model_path.exists() {
        anyhow::bail!("model not exists");
    }

    let device = &Device::Cpu;

    let data_dir = &Path::new("gpt2/gpt2/355M");
    let settings = {
        let (mut s, _) = gpt2::load_settings_and_params(&data_dir).expect("load gpt2 config");
        s.drop_rate = 0.0;
        s
    };

    let model = GptModel::<B>::new(&settings, device);

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = &model.load_file(model_path, &recorder, device).context("load model")?;

    B::seed(123);
    let tokenizer = Encoding::gpt2();

    let (_, test_data, _) = utils::load_and_split_data("instruction-data.json").context("load and split data")?;

    let pad_token_id = Some(chapter07::PAD_TOKEN_ID as usize);
    for (i, entry) in test_data.iter().enumerate().take(3) {
        let input_text = utils::format_input(entry);

        let idx = tokenizer.tokenize(&input_text).to_device(device);

        let token_ids = chapter05::utils::generate(model, idx, 256, settings.context_length, None, None, pad_token_id);

        let generated_text = tokenizer
            .detokenize(token_ids)
            .with_context(|| format!("detokenize {i}-th output"))?;

        let response_text = generated_text
            .split_at(input_text.len())
            .1
            .trim()
            .replace("### Response:", "");

        println!("{input_text}");
        println!("\nCorrect Response:\n>> {}", entry.output);
        // 再次 trim 是因为 "### Response:" 后紧跟了一个换行符
        println!("\nModel Response:\n>> {}", response_text.trim());
        println!("--------------------------------------");
    }

    // Listing 7.9 Generating test set responses
    let p = ProgressBar::new(test_data.len() as u64);
    let mut out = Vec::with_capacity(test_data.len());
    for (i, entry) in test_data.into_iter().enumerate() {
        let input_text = utils::format_input(&entry);

        let idx = tokenizer.tokenize(&input_text).to_device(device);

        let token_ids = chapter05::utils::generate(model, idx, 256, settings.context_length, None, None, pad_token_id);

        let generated_text = tokenizer
            .detokenize(token_ids)
            .with_context(|| format!("detokenize {i}-th output"))?;

        let response_text = generated_text
            .split_at(input_text.len())
            .1
            .replace("### Response:", "")
            .trim()
            .to_owned();

        let o = DataWithModelResponse {
            data: entry,
            model_response: response_text,
        };
        out.push(o);

        p.inc(1);
    }
    p.finish();

    let out_path = Path::new("instruction-data-with-response.json");
    let mut f = File::create(out_path).context("create output file")?;
    serde_json::to_writer_pretty(&mut f, &out).context("json write")?;

    println!("{:?}", out[0]);

    Ok(())
}
