use chapter07::PAD_TOKEN_ID;
use tiktoken::ext::Encoding;

fn main() -> anyhow::Result<()> {
    let tokenizer = Encoding::gpt2();

    let allowed_special = ["<|endoftext|>"].into();
    let got = tokenizer.encode("<|endoftext|>", &allowed_special);
    assert_eq!(PAD_TOKEN_ID, got[0], "unexpected token id for <|endoftext|>");

    Ok(())
}
