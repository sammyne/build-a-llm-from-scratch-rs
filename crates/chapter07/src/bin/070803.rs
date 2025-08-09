use anyhow::Context;
use chapter07::ollama;
use chapter07::utils::{self, DataWithModelResponse};
use clap::Parser;
use indicatif::ProgressIterator;
use reqwest::Url;

/// 准备步骤
/// 1. 运行 ollama：docker run -it --rm -v $PWD/_ollama:/root/.ollama --name ollama ollama/ollama:0.11.4 serve
/// 2. 使用 8B 的 Llama 3 模型：docker exec -it ollama ollama run llama3
/// 3. 查询 ollama 服务地址：docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ollama
/// 4. 将上述地址设置为下述程序的 url 选项
fn main() -> anyhow::Result<()> {
    let Cli { model, url } = Cli::parse();
    let model = model.as_str();

    let test_data = utils::load_json::<_, DataWithModelResponse>("instruction-data-with-response.json")
        .context("load test data")?;

    let scores = generate_model_scores(&test_data, model, url)?;
    println!("Number of scores: {} of {}", scores.len(), test_data.len());

    println!(
        "Average score: {:.2}\n",
        scores.iter().map(|&v| v as f32).sum::<f32>() / scores.len() as f32
    );

    Ok(())
}

#[derive(Parser)]
pub struct Cli {
    /// 模型名称。
    #[clap(long, default_value = "llama3")]
    pub model: String,
    /// ollama 服务器的 URL。
    #[clap(long, default_value = "http://172.17.0.6:11434/api/chat")]
    pub url: Url,
}

fn generate_model_scores(json_data: &[DataWithModelResponse], model: &str, url: Url) -> anyhow::Result<Vec<u8>> {
    println!("Scoring entries");

    let mut scores = Vec::with_capacity(json_data.len());
    for (i, entry) in json_data.iter().enumerate().progress() {
        let prompt = format!(
            "Given the input `{}` and correct output `{}`, score the model response `{}` on a scale from 0 to 100, where 100 is the best score. Respond with the integer number only.",
            utils::format_input(entry),
            entry.output,
            entry.model_response
        );

        let score: u8 = ollama::query_model(&prompt, model, url.clone())
            .with_context(|| format!("{i}-th query_model"))?
            .parse()
            .with_context(|| format!("{i}-th parse score"))?;

        scores.push(score);
    }

    Ok(scores)
}
