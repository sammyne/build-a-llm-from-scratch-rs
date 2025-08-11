use std::collections::BTreeSet;

use anyhow::Context;
use chapter02::verdict;

fn main() -> anyhow::Result<()> {
    let vocab: BTreeSet<String> = verdict::load_and_canonicalize().context("load-and-canonicalize verdict")?;

    println!("{}", vocab.len());

    Ok(())
}
