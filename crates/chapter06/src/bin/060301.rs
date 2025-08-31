use std::collections::HashSet;

use tiktoken::ext::Encoding;

fn main() -> anyhow::Result<()> {
    let tokenizer = Encoding::gpt2();

    let allowed_specials = ["<|endoftext|>"].into_iter().collect::<HashSet<_>>();
    let got = tokenizer.encode("<|endoftext|>", &allowed_specials);

    let expect = vec![50256u32];
    assert_eq!(expect, got);

    Ok(())
}
