use std::collections::HashMap;

use anyhow::Context;
use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::tensor::{DType, Tensor, activation};
use plotters::prelude::*;

type B = NdArray;

type D = <B as Backend>::Device;

/// TODO(xiangminli): plotters 绘制并排的直方图挺难搞，换个绘图库。
fn main() -> anyhow::Result<()> {
    B::seed(123);
    let device = &D::Cpu;

    let vocab = [
        ("closer", 0i64),
        ("every", 1),
        ("effort", 2),
        ("forward", 3),
        ("inches", 4),
        ("moves", 5),
        ("pizza", 6),
        ("toward", 7),
        ("you", 8),
    ]
    .into_iter()
    .collect::<HashMap<&str, i64>>();

    let inverse_vocab: HashMap<_, _> = vocab.iter().map(|(&k, &v)| (v, k)).collect();

    let next_token_logits =
        Tensor::<B, 1>::from_floats([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79], device);

    let probas = activation::softmax(next_token_logits.clone(), 0);

    // for t in [1,0.1,dd]

    let scaled_probas = softmax_with_temperature(probas.clone(), 1.0)
        .to_data()
        .convert_dtype(DType::F32)
        .to_vec::<f32>()
        .map_err(|err| anyhow::anyhow!("tensor as vec {err:?}"))?;

    // let x:Vec<usize> = (0..vocab.len()).collect();

    plot(scaled_probas, &inverse_vocab, 1.0).context("plot")?;

    Ok(())
}

fn softmax_with_temperature(logits: Tensor<B, 1>, temperature: f32) -> Tensor<B, 1> {
    let scaled = logits / temperature;
    activation::softmax(scaled, 0)
}

fn plot(probas: Vec<f32>, vocab: &HashMap<i64, &'static str>, temperature: f32) -> anyhow::Result<()> {
    // 创建绘图区域，大小为 500x300 像素
    let root = SVGBackend::new("visualize-temperature-effect.svg", (500, 300)).into_drawing_area();
    root.fill(&WHITE)?;

    // 计算坐标轴范围
    let max_prob = probas.iter().fold(0.0f32, |a, &b| a.max(b));

    // 创建图表构建器
    let mut chart = ChartBuilder::on(&root)
        .caption("Temperature Effect on Softmax", ("sans-serif", 15))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(30)
        .build_cartesian_2d((0..vocab.len()).into_segmented(), 0.0..max_prob * 1.1)?;

    // 配置网格和坐标轴
    chart
        .configure_mesh()
        .x_labels(vocab.len())
        .y_labels(5)
        .x_desc("Vocabulary")
        .y_desc("Probability")
        .x_label_formatter(&|v| match v {
            SegmentValue::CenterOf(i) if *i < vocab.len() => vocab[&(*i as i64)].to_string(),
            _ => "".to_string(),
        })
        .draw()?;

    // 绘制不同温度下的概率分布
    chart
        .draw_series(
            Histogram::vertical(&chart)
                .data(vocab.iter().enumerate().map(|(j, _)| (j, probas[j])))
                .style(RED.filled())
                .margin(2),
            // .width(bar_width)
            // .offset(offset[i]),
        )?
        .label(format!("T = {:.1}", temperature))
        .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], RED));

    // 添加图例
    chart
        .configure_series_labels()
        .background_style(&WHITE)
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    // 保存图表
    root.present()?;

    Ok(())
}
