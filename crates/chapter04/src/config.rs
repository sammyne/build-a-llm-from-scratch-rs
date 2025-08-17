use std::sync::LazyLock;

/// 默认值采用 GPT-124M 的配置。
#[derive(burn::prelude::Config, Copy, Debug)]
pub struct Config {
    #[config(default = 50257)]
    pub vocab_size: usize,
    #[config(default = 1024)]
    pub context_length: usize,
    #[config(default = 768)]
    pub emb_dim: usize,
    #[config(default = 12)]
    pub nheads: usize,
    #[config(default = 12)]
    pub nlayers: usize,
    #[config(default = 0.1)]
    pub drop_rate: f64,
    #[config(default = false)]
    pub qkv_bias: bool,
}

pub static GPT_124M: LazyLock<Config> = LazyLock::new(Config::new);
