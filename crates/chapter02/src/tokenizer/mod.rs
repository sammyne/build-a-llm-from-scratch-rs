mod v1;
mod v2;

pub use v1::TokenizerV1;
pub use v2::TokenizerV2;

pub const TOKEN_ENDOFTEXT: &str = "<|endoftext|>";
pub const TOKEN_UNKNOWN: &str = "<|unk|>";

pub fn extend_with_unknown_and_endoftext<T: Extend<String>>(mut v: T) -> T {
    v.extend([TOKEN_ENDOFTEXT, TOKEN_UNKNOWN].map(|v| v.to_owned()));
    v
}
