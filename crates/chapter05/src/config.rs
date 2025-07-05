use chapter04::Config;

pub static GPT_124M: &Config = &Config {
    vocab_size: 50257,
    context_length: 256,
    emb_dim: 768,
    nheads: 12,
    nlayers: 12,
    drop_rate: 0.1,
    qkv_bias: false,
};
