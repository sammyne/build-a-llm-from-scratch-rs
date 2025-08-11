use std::collections::BTreeSet;

use anyhow::Context;
use chapter02::tokenizer::SimpleTokenizerV1;
use chapter02::verdict;

fn main() -> anyhow::Result<()> {
    let vocab: BTreeSet<String> = verdict::load_and_canonicalize().context("load-and-canonicalize verdict")?;

    let tokenizer = SimpleTokenizerV1::new(vocab);

    let text = r#""It's the last he painted, you know,"Mrs. Gisburn said with pardonable pride."#;
    let ids = tokenizer.encode(text);
    println!("{ids:?}");
    println!("{}", tokenizer.decode(&ids));

    Ok(())
}
