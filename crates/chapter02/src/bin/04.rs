use std::collections::BTreeSet;

use anyhow::Context;
use chapter02::tokenizer::{self, TokenizerV2};
use chapter02::verdict;

fn main() -> anyhow::Result<()> {
    let vocab: Vec<_> = verdict::load_and_canonicalize::<BTreeSet<_>>()
        .context("load verdict")?
        .into_iter()
        .collect();
    let vocab = tokenizer::extend_with_unknown_and_endoftext(vocab);
    println!("{}", vocab.len());
    for (i, v) in vocab.iter().enumerate().skip(vocab.len() - 5) {
        println!("({v}, {i})");
    }

    let text1 = "Hello, do you like tea?";
    let text2 = "In the sunlit terraces of the palace.";
    let text = [text1, text2].join(" <|endoftext|> ");
    println!("{text}");

    let tokenizer = TokenizerV2::new(vocab);
    println!("{:?}", tokenizer.encode(&text));
    println!("{}", tokenizer.decode(&tokenizer.encode(&text)));

    Ok(())
}
