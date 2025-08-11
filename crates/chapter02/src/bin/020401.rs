use std::collections::BTreeSet;

use anyhow::Context;
use chapter02::tokenizer::SimpleTokenizerV1;
use chapter02::verdict;

fn main() -> anyhow::Result<()> {
    let vocab: BTreeSet<String> = verdict::load_and_canonicalize().context("load-and-canonicalize verdict")?;

    let tokenizer = SimpleTokenizerV1::new(vocab);

    let text = r#"Hello, do you like tea. Is this-- a test?"#;
    let _ids = tokenizer.encode(text);

    Ok(())
}
