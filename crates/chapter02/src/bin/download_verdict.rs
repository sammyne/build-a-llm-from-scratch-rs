use std::fs::File;
use std::io::copy;

use reqwest::blocking;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const URL: &str = std::concat!(
        "https://raw.githubusercontent.com/rasbt/",
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/",
        "the-verdict.txt"
    );
    let response = blocking::get(URL)?;
    let mut dest = File::create("static/the-verdict.txt")?;
    copy(&mut response.bytes()?.as_ref(), &mut dest)?;
    Ok(())
}
