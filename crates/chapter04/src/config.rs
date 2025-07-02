use std::sync::LazyLock;

pub struct Config {
    pub vocab_size: usize,
    pub context_length: usize,
    pub emb_dim: usize,
    pub nheads: usize,
    pub nlayers: usize,
    pub drop_rate: f64,
    pub pkv_bias: bool,
}

pub static GPT_124M: LazyLock<Config> = LazyLock::new(|| Config {
    vocab_size: 50257,
    context_length: 1024,
    emb_dim: 768,
    nheads: 12,
    nlayers: 12,
    drop_rate: 0.1,
    pkv_bias: false,
});
