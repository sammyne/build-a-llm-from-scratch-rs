use anyhow::Context;
use regex::Regex;

fn main() -> anyhow::Result<()> {
    let text = "Hello, world. This, is a test.";

    let r = Regex::new(r#"([,.]|\s)"#).context("build regex")?;
    let result = chapter02::strings::split(text, Some(r));

    println!("{result:?}");

    Ok(())
}
