use anyhow::Context;
use chapter02::verdict;
use regex::Regex;

fn main() -> anyhow::Result<()> {
    let text = "Hello, world. Is this-- a test?";

    let result = must_split(text);

    let result: Vec<&str> = result.into_iter().filter(|s| !s.trim().is_empty()).collect();
    println!("{:?}", result);

    let raw_text = verdict::load().context("load verdict")?;
    let preprocessed: Vec<_> = must_split(&raw_text)
        .into_iter()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
        .collect();
    println!("len(preprocessed) = {}", preprocessed.len());
    println!("preprocessed[:30] = {:?}", &preprocessed[0..30]);

    Ok(())
}

fn must_split(text: &str) -> Vec<&str> {
    let re = Regex::new(r#"([,.:;?_!"()'\\]|--|\s)"#).unwrap();
    let mut result = Vec::new();
    let mut last = 0;

    for cap in re.captures_iter(text) {
        let m = cap.get(0).unwrap();
        if m.start() > last {
            result.push(&text[last..m.start()]);
        }
        result.push(m.as_str());
        last = m.end();
    }

    if last < text.len() {
        result.push(&text[last..]);
    }

    result
}
