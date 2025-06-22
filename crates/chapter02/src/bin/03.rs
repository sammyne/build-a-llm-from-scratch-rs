use std::collections::BTreeSet;

use anyhow::Context;
use chapter02::tokenizer::TokenizerV1;
use chapter02::verdict;

fn main() -> anyhow::Result<()> {
    let vocab: BTreeSet<String> = verdict::load_and_canonicalize().context("load-and-canonicalize verdict")?;

    let tokenizer = TokenizerV1::new(vocab);

    let text = r#""It's the last he painted, you know,"Mrs. Gisburn said with pardonable pride."#;
    let ids = tokenizer.encode(text);
    println!("{ids:?}");
    println!("{}", tokenizer.decode(&ids));

    let text = "Hello, do you like tea?";
    println!("{:?}", tokenizer.encode(text));

    Ok(())
}
