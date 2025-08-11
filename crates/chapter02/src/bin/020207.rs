use anyhow::Context;
use chapter02::verdict;
use regex::Regex;

fn main() -> anyhow::Result<()> {
    let raw_text = verdict::load().context("load verdict")?;
    let r = Regex::new(r#"([,.:;?_!"()']|--|\s)"#).context("build regex")?;

    let preprocessed: Vec<_> = chapter02::strings::split(&raw_text, Some(r))
        .into_iter()
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
        .collect();
    println!("preprocessed[:30] = {:?}", &preprocessed[0..30]);
    println!("len(preprocessed) = {}", preprocessed.len());

    Ok(())
}
