use anyhow::Context as _;
use reqwest::Url;
use reqwest::blocking::Client;
use serde::Deserialize;

/// Listing 7.10 Querying a local Ollama model
pub fn query_model(prompt: &str, model: &str, url: Url) -> anyhow::Result<String> {
    let data = serde_json::json!({
      "model": model,
      "messages": [
        {
          "role": "user",
          "content": prompt,
        }
      ],
      "options": {
        "seed": 123,
        "temperature": 0,
        "num_ctx": 2048,
      }
    });

    let payload = serde_json::to_string(&data).context("json encode payload")?;

    let response = Client::new()
        .post(url)
        .body(payload)
        .header("Content-Type", "application/json")
        .send()
        .context("send request")?;

    #[derive(Deserialize)]
    struct Message {
        pub content: String,
    }

    #[derive(Deserialize)]
    struct Line {
        pub message: Message,
    }

    let mut response_data = String::new();
    for line in response.text().context("read response text")?.lines() {
        let b: Line = serde_json::from_str(line).with_context(|| format!("decode line: {line}"))?;
        response_data += &b.message.content;
    }

    Ok(response_data)
}
