use std::fs::File;
use std::path::Path;

use anyhow::Context as _;
use burn::backend::LibTorch;
use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use chapter05::utils::{GenerateOptions, Tokenizer};
use chapter07::utils::{self, DataWithModelResponse};
use indicatif::ProgressBar;
use tiktoken::ext::Encoding;

type B = LibTorch;

type Device = <B as Backend>::Device;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new("gpt-355m-model-sft.mpk");
    if !model_path.exists() {
        anyhow::bail!("model not exists");
    }

    let device = &Device::Cpu;

    let model = chapter06::utils::load_gpt2::<B, _>("gpt2/355M", device).context("load GPT-2")?;

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = &model.load_file(model_path, &recorder, device).context("load model")?;

    B::seed(123);
    let tokenizer = Encoding::gpt2();

    let (_, test_data, _) = utils::load_and_split_data("instruction-data.json").context("load and split data")?;

    // Listing 7.9 Generating test set responses
    let p = ProgressBar::new(test_data.len() as u64);
    let mut out = Vec::with_capacity(test_data.len());
    let opts =
        GenerateOptions::new(256, model.pos_emb.weight.dims()[0]).with_eos_id(Some(chapter07::PAD_TOKEN_ID as usize));
    for (i, entry) in test_data.into_iter().enumerate() {
        let input_text = utils::format_input(&entry);

        let idx = tokenizer.tokenize(&input_text).to_device(device);

        let token_ids = chapter05::utils::generate(model, idx, opts);

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
