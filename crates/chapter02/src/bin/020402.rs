use std::collections::BTreeSet;

use anyhow::Context;
use chapter02::{tokenizer, verdict};

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

    Ok(())
}
