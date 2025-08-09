use anyhow::Context;
use chapter07::ollama;
use chapter07::utils::{self, DataWithModelResponse};
use clap::Parser;
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

    for (i, entry) in test_data.iter().enumerate().take(3) {
        let prompt = format!(
            "Given the input `{}` and correct output `{}`, score the model response `{}` on a scale from 0 to 100, where 100 is the best score. ",
            utils::format_input(entry),
            entry.output,
            entry.model_response
        );

        let r = ollama::query_model(&prompt, model, url.clone()).with_context(|| format!("{i}-th query model"))?;

        println!("\nDataset response:");
        println!(">> {}", entry.output);
        println!("\nModel response:");
        println!(">> {}", entry.model_response);
        println!("\nScore:");
        println!(">> {r}");
        println!("\n-------------------------")
    }

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
