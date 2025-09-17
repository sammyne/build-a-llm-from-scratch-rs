use std::fs;
use std::path::Path;

use anyhow::Context;

fn main() -> anyhow::Result<()> {
    const FILE_PATH: &str = "instruction-data.json";
    const URL: &str = std::concat!(
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch",
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    );

    let path: &Path = FILE_PATH.as_ref();
    if path.exists() {
        return Ok(());
    }

    let r = reqwest::blocking::get(URL).context("http get")?;
    if !r.status().is_success() {
        anyhow::bail!("bad http status: {}", r.status());
    }

    let body = r.text().context("read http response body")?;
    fs::write(path, body).context("save the downloaded file")?;

    Ok(())
}
