use std::collections::BTreeSet;

use anyhow::Context;
use chapter02::verdict;

fn main() -> anyhow::Result<()> {
    let vocab: BTreeSet<String> = verdict::load_and_canonicalize().context("load-and-canonicalize verdict")?;
    for (i, v) in vocab.iter().enumerate().take(51) {
        println!("({v}, {i})");
    }

    Ok(())
}
