use anyhow::Context;
use chapter02::verdict;
use tiktoken::ext::Encoding;

fn main() -> anyhow::Result<()> {
    let raw_text = verdict::load().context("load verdict")?;

    let tokenizer = Encoding::gpt2();

    let enc_text = tokenizer.encode(&raw_text, &Default::default());
    println!("{}", enc_text.len());

    let enc_sample = &enc_text[50..];

    const CONTEXT_SIZE: usize = 4;
    let x = &enc_sample[0..CONTEXT_SIZE];
    let y = &enc_sample[1..CONTEXT_SIZE + 1];
    println!();
    println!("x: {x:?}");
    println!("y:      {y:?}");

    println!();
    for i in 1..CONTEXT_SIZE + 1 {
        let context = &enc_sample[..i];
        let desired = enc_sample[i];
        println!("{context:?} ----> {desired}");
    }

    println!();
    for i in 1..CONTEXT_SIZE + 1 {
        let context = &enc_sample[..i];
        let desired = enc_sample[i];

        let context = tokenizer.decode_str(context).context("decode context")?;
        let desired = tokenizer.decode_str(&[desired]).context("decode desired")?;
        println!("{context} ----> {desired}");
    }

    Ok(())
}
