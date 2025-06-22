use std::collections::HashSet;

use anyhow::Context;
use chapter02::tokenizer::TOKEN_ENDOFTEXT;
use tiktoken::ext::Encoding;

fn main() -> anyhow::Result<()> {
    let tokenizer = Encoding::gpt2();

    let allowed_special: HashSet<_> = [TOKEN_ENDOFTEXT].into_iter().collect();

    let text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.";

    let ids = tokenizer.encode(text, &allowed_special);
    let expect = vec![
        15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13,
    ];
    assert_eq!(expect, ids, "unexpected ids");

    let decoded = tokenizer.decode_str(&ids).context("decode")?;
    let expect = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.";
    assert_eq!(expect, decoded, "unexpected decoded");

    Ok(())
}
