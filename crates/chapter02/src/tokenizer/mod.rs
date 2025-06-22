mod v1;
mod v2;

use tiktoken::ext::Encoding;
pub use v1::TokenizerV1;
pub use v2::TokenizerV2;

pub const TOKEN_ENDOFTEXT: &str = "<|endoftext|>";
pub const TOKEN_UNKNOWN: &str = "<|unk|>";

pub trait Tokenizer {
    fn decode(&self, ids: &[u32]) -> anyhow::Result<String>;

    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>>;
}

impl Tokenizer for Encoding {
    fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.decode_str(ids)
    }

    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let out = self.encode(text, &Default::default());
        Ok(out)
    }
}

pub fn extend_with_unknown_and_endoftext<T: Extend<String>>(mut v: T) -> T {
    v.extend([TOKEN_ENDOFTEXT, TOKEN_UNKNOWN].map(|v| v.to_owned()));
    v
}
