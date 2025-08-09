use anyhow::Context;
use chapter07::ollama;
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

    // 快速验证
    let result = ollama::query_model("What do Llamas eat?", &model, url.clone()).context("quickcheck query_model")?;
    println!("{result}");

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
