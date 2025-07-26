use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;

use anyhow::Context as _;
use burn::backend::LibTorch;
use burn::module::{Module, Param};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use chapter04::{Config, GptModel};
use chapter05::config::GPT_124M;
use chapter05::utils::{self, Tokenizer as _};
use tiktoken::ext::Encoding;

type B = LibTorch;

type Device = <LibTorch as Backend>::Device;

/// 需要先进去 gpt2 运行 uv run main.py 准备好数据。
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    let (settings, params) = load_gpt2().expect("load gpt2 config");

    let mut model = GptModel::<B>::new(&settings, device);

    load_weights_into_gpt2(params, &mut model).context("load weights into model")?;

    let tokenizer = Encoding::gpt2();
    let idx = tokenizer.tokenize("Every effort moves you").to_device(device);

    B::seed(123);

    let temperature = 1.5.into();
    let token_ids = utils::generate(&model, idx, 25, settings.context_length, temperature, Some(50), None);
    let out = tokenizer.detokenize(token_ids).context("decode output")?;
    println!("Output text:\n{out}");

    Ok(())
}

// Settings: {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
// Parameter dictionary keys: dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])
// [[-0.11010301 -0.03926672  0.03310751 ... -0.1363697   0.01506208
//    0.04531523]
//  [ 0.04034033 -0.04861503  0.04624869 ...  0.08605453  0.00253983
//    0.04318958]
//  [-0.12746179  0.04793796  0.18410145 ...  0.08991534 -0.12972379
//   -0.08785918]
//  ...
//  [-0.04453601 -0.05483596  0.01225674 ...  0.10435229  0.09783269
//   -0.06952604]
//  [ 0.1860082   0.01665728  0.04611587 ... -0.09625227  0.07847701
//   -0.02245961]
//  [ 0.05135201 -0.02768905  0.0499369  ...  0.00704835  0.15519823
//    0.12067825]]
// Token embedding weight tensor dimensions: (50257, 768)
#[derive(Debug, serde::Deserialize)]
struct Gpt2Params {
    pub blocks: Vec<Block>,
    pub g: Vec<f32>,
    pub b: Vec<f32>,
    pub wte: Vec<Vec<f32>>,
    pub wpe: Vec<Vec<f32>>,
}

#[derive(Debug, serde::Deserialize)]
struct Block {
    pub attn: Attn,       // Attention weights
    pub mlp: FeedForward, // Feed-forward network weights
    pub ln_1: LayerNorm,
    pub ln_2: LayerNorm,
}

#[derive(Debug, serde::Deserialize)]
struct Attn {
    pub c_attn: AttnQkv,
    pub c_proj: AttnOutProj, // Projection weights for attention
}

#[derive(Debug, serde::Deserialize)]
pub struct AttnQkv {
    pub w: Vec<Vec<f32>>, // Weights for the attention layer
    pub b: Vec<f32>,      // Bias for the attention layer
}

#[derive(Debug, serde::Deserialize)]
pub struct AttnOutProj {
    pub w: Vec<Vec<f32>>, // Weights for the output projection
    pub b: Vec<f32>,      // Bias for the output projection
}

#[derive(Debug, serde::Deserialize)]
pub struct FeedForward {
    pub c_fc: FeedForwardWb,
    pub c_proj: FeedForwardWb,
}

#[derive(Debug, serde::Deserialize)]
pub struct FeedForwardWb {
    pub w: Vec<Vec<f32>>, // Weights for the feed-forward layer
    pub b: Vec<f32>,      // Bias for the feed-forward layer
}

#[derive(Debug, serde::Deserialize)]
pub struct LayerNorm {
    pub g: Vec<f32>, // Weights for the feed-forward layer
    pub b: Vec<f32>, // Bias for the feed-forward layer
}

fn checked_assign_1d_param(param: &mut Param<Tensor<B, 1>>, value: &[f32]) -> anyhow::Result<()> {
    let value = checked_new_1d_like(value, &param.val()).context("tensor-ize data")?;
    checked_assign_param(param, value).context("assign")
}

fn checked_assign_2d_param(
    param: &mut Param<Tensor<B, 2>>,
    value: &[Vec<f32>],
    transposed: bool,
) -> anyhow::Result<()> {
    if transposed {
        renew_param(param, |v| {
            checked_new_2d_like(&value, &v.clone().transpose()).map(|v| v.transpose())
        })
    } else {
        renew_param(param, |v| checked_new_2d_like(&value, &v.clone()))
    }
}

fn checked_assign_param<const D: usize>(param: &mut Param<Tensor<B, D>>, value: Tensor<B, D>) -> anyhow::Result<()> {
    let expected_shape = param.val().shape().dims;
    let actual_shape = value.shape().dims;

    if expected_shape != actual_shape {
        anyhow::bail!("shape mismatch: expect {expected_shape:?}, got {actual_shape:?}");
    }

    *param = Param::initialized(param.id, value);

    Ok(())
}

fn checked_new_1d_like(data: &[f32], like: &Tensor<B, 1>) -> anyhow::Result<Tensor<B, 1>> {
    let device = &like.device();

    let expected_len = like.shape().dims[0];
    if data.len() != expected_len {
        anyhow::bail!("Expected length {}, but got {}", expected_len, data.len());
    }

    let out = Tensor::from_floats(data, device);
    Ok(out)
}

fn checked_new_2d_like(data: &[Vec<f32>], like: &Tensor<B, 2>) -> anyhow::Result<Tensor<B, 2>> {
    let device = &like.device();

    let (row, col) = {
        let dims = like.shape().dims;
        (dims[0], dims[1])
    };

    anyhow::ensure!(row == data.len(), "bad #(rows): expect {}, got {}", row, data.len());

    let mut stacked = Vec::with_capacity(data.len());
    for (i, v) in data.iter().enumerate() {
        anyhow::ensure!(
            v.len() == col,
            "bad #(cols) of {}-th row: expect {}, got {}",
            i,
            col,
            data[i].len()
        );

        let v = Tensor::<B, 1>::from_floats(v.as_slice(), device);
        stacked.push(v);
    }

    let out = Tensor::stack::<2>(stacked, 0);
    Ok(out)
}

fn tripple_split_1d(data: &[f32], device: &Device) -> anyhow::Result<(Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>)> {
    anyhow::ensure!(data.len() % 3 == 0, "bad len = {}, not multiple of 3", data.len(),);

    let t = Tensor::from_floats(data, device);

    let splits = t.split(data.len() / 3, 0);

    Ok((splits[0].clone(), splits[1].clone(), splits[2].clone()))
}

fn tripple_split_2d(data: &[Vec<f32>], device: &Device) -> anyhow::Result<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>)> {
    anyhow::ensure!(
        data[0].len() % 3 == 0,
        "bad #(cols)={} of 1st row, not multiple of 3",
        data[0].len(),
    );

    let mut stacked = Vec::with_capacity(data.len());

    let expected_cols = data[0].len();
    for (i, v) in data.iter().enumerate() {
        anyhow::ensure!(
            v.len() == expected_cols,
            "bad #(cols) of {}-th row, expect {}, got {}",
            i,
            expected_cols,
            v.len(),
        );

        let v = Tensor::<B, 1>::from_floats(v.as_slice(), device);
        stacked.push(v);
    }

    let t = Tensor::stack::<2>(stacked, 0);

    let splits = t.split(expected_cols / 3, 1);

    Ok((splits[0].clone(), splits[1].clone(), splits[2].clone()))
}

fn renew_param<const D: usize, F>(param: &mut Param<Tensor<B, D>>, f: F) -> anyhow::Result<()>
where
    F: FnOnce(&Tensor<B, D>) -> anyhow::Result<Tensor<B, D>>,
{
    let v = f(&param.val()).context("transform value")?;
    *param = Param::initialized(param.id, v);
    Ok(())
}

fn load_gpt2() -> anyhow::Result<(Config, Gpt2Params)> {
    let data_dir = Path::new("gpt2/gpt2/124M");

    let p = data_dir.join("hparams.json");
    if !p.exists() {
        return Err(anyhow::anyhow!("GPT-2 config file not found at {p:?}"));
    }
    let c = load_gpt2_config(&p).context("load config")?;

    let p = data_dir.join("params-124m.json");
    if !p.exists() {
        return Err(anyhow::anyhow!("GPT-2 params file not found at {p:?}"));
    }
    let p = load_gpt2_params(&p).context("load params")?;

    Ok((c, p))
}

fn load_gpt2_config(p: &Path) -> anyhow::Result<Config> {
    let json = fs::read_to_string(p).context("read file")?;

    #[derive(serde::Deserialize)]
    struct Gpt2Config {
        n_vocab: usize,
        n_ctx: usize,
        n_embd: usize,
        n_head: usize,
        n_layer: usize,
    }

    let gpt2: Gpt2Config = serde_json::from_str(&json).context("json decode")?;

    let mut out = GPT_124M.clone();
    out.vocab_size = gpt2.n_vocab;
    out.context_length = gpt2.n_ctx;
    out.emb_dim = gpt2.n_embd;
    out.nheads = gpt2.n_head;
    out.nlayers = gpt2.n_layer;

    out.qkv_bias = true; // GPT-2 uses bias in QKV layers

    Ok(out)
}

fn load_gpt2_params(p: &Path) -> anyhow::Result<Gpt2Params> {
    let f = File::open(p).context("open file")?;

    serde_json::from_reader(BufReader::new(f)).context("json decode")
    // serde_json::from_str(&json).context("json decode")
}

fn load_weights_into_gpt2(params: Gpt2Params, model: &mut GptModel<B>) -> anyhow::Result<()> {
    let device = &model.devices()[0].clone();

    checked_assign_2d_param(&mut model.pos_emb.weight, &params.wpe, false).context("load positional embeddings")?;

    checked_assign_2d_param(&mut model.tok_emb.weight, &params.wte, false).context("load token embeddings")?;

    anyhow::ensure!(
        model.trf_blocks.len() == params.blocks.len(),
        "model has {} transformer blocks, but params has {}",
        model.trf_blocks.len(),
        params.blocks.len()
    );
    for (dst, src) in model.trf_blocks.iter_mut().zip(params.blocks.iter()) {
        let (q_w, k_w, v_w) = tripple_split_2d(&src.attn.c_attn.w, &device)?;
        checked_assign_param(&mut dst.attn.q.weight, q_w).context("load attention query weights")?;
        checked_assign_param(&mut dst.attn.k.weight, k_w).context("load attention key weights")?;
        checked_assign_param(&mut dst.attn.v.weight, v_w).context("load attention value weights")?;

        let (q_b, k_b, v_b) = tripple_split_1d(&src.attn.c_attn.b, &device)?;
        checked_assign_param(dst.attn.q.bias.as_mut().expect("miss q-bias"), q_b)
            .context("load attention query bias")?;
        checked_assign_param(dst.attn.k.bias.as_mut().expect("miss k-bias"), k_b).context("load attention key bias")?;
        checked_assign_param(dst.attn.v.bias.as_mut().expect("miss v-bias"), v_b)
            .context("load attention value bias")?;

        // pytorch 的线性层存的是转置，burn 存的是原始值。
        checked_assign_2d_param(&mut dst.attn.out_proj.weight, &src.attn.c_proj.w, false)
            .context("load attention out-proj weights")?;
        let b = dst.attn.out_proj.bias.as_mut().expect("miss out-proj bias");
        checked_assign_1d_param(b, &src.attn.c_proj.b).context("load out-proj bias")?;

        checked_assign_2d_param(&mut dst.ff.linear1.weight, &src.mlp.c_fc.w, false)
            .context("load feed-forward linear1 weights")?;
        let b = dst.ff.linear1.bias.as_mut().expect("miss ff.linear1 bias");
        checked_assign_1d_param(b, &src.mlp.c_fc.b).context("load ff.linear1 bias")?;
        checked_assign_2d_param(&mut dst.ff.linear2.weight, &src.mlp.c_proj.w, false)
            .context("load feed-forward linear2 weights")?;
        let b = dst.ff.linear2.bias.as_mut().expect("miss ff.linear2 bias");
        checked_assign_1d_param(b, &src.mlp.c_proj.b).context("load ff.linear2 bias")?;

        checked_assign_1d_param(&mut dst.norm1.scale, &src.ln_1.g).context("load norm1.scale")?;
        checked_assign_1d_param(&mut dst.norm1.shift, &src.ln_1.b).context("load norm1.shift")?;
        checked_assign_1d_param(&mut dst.norm2.scale, &src.ln_2.g).context("load norm2.scale")?;
        checked_assign_1d_param(&mut dst.norm2.shift, &src.ln_2.b).context("load norm2.shift")?;
    }

    checked_assign_1d_param(&mut model.final_norm.scale, &params.g).context("load final_norm.scale")?;
    checked_assign_1d_param(&mut model.final_norm.shift, &params.b).context("load final_norm.shift")?;
    checked_assign_2d_param(&mut model.out_head.weight, &params.wte, true).context("load out_head weights")?;

    Ok(())
}
