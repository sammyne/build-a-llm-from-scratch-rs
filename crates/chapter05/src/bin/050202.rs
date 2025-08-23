use std::fs::File;

use anyhow::Context as _;
use plotters::element::DashedPathElement;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> anyhow::Result<()> {
    const PATH: &str = "train-overview.json";
    const OUT_PATH: &str = "loss.svg";

    let data = TrainOveriew::load(PATH).context("load train overview")?;

    plot(data, OUT_PATH).context("plot")?;

    println!("已将数据绘制到 {OUT_PATH}");

    Ok(())
}

#[derive(Serialize, Deserialize)]
struct TrainOveriew {
    epoches: usize,
    train_losses: Vec<f32>,
    val_losses: Vec<f32>,
    track_tokens_seen: Vec<usize>,
}

impl TrainOveriew {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let mut f = File::open(path).context("open file")?;
        serde_json::from_reader(&mut f).context("json loads")
    }
}

fn plot(data: TrainOveriew, path: &str) -> anyhow::Result<()> {
    let TrainOveriew {
        epoches,
        train_losses,
        val_losses,
        ..
    } = data;

    let x: Vec<f32> = {
        let s = (epoches as f32) / ((train_losses.len() - 1) as f32);
        (0..=train_losses.len()).map(|v| v as f32 * s).collect()
    };

    // 创建绘图区域，大小为 500x300 像素，保存为 SVG。
    let root = SVGBackend::new(path, (500, 300)).into_drawing_area();
    root.fill(&WHITE).context("fill drawing area")?;

    // 创建主图表
    let y_axis = {
        let m = train_losses
            .iter()
            .chain(val_losses.iter())
            .copied()
            .fold(0.0f32, |acc, v| acc.max(v));
        0.0..m
    };

    let mut chart = ChartBuilder::on(&root)
        .caption("Training and Validation Loss", ("sans-serif", 10))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..(x[x.len() - 1]), y_axis.clone())?;

    // 配置网格和坐标轴
    chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_desc("Epochs")
        .y_desc("Loss")
        .draw()?;

    // 绘制训练损失曲线
    chart
        .draw_series(LineSeries::new(
            x.iter().copied().zip(train_losses),
            BLUE.stroke_width(2),
        ))
        .context("draw training loss")?
        .label("Training loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // 绘制验证损失曲线（虚线样式）
    chart
        .draw_series(DashedLineSeries::new(
            x.iter().copied().zip(val_losses),
            10,
            5,
            RED.stroke_width(2),
        ))
        .context("draw validation loss")?
        .label("Validation loss")
        .legend(|(x, y)| DashedPathElement::new(vec![(x, y), (x + 20, y)], 10, 5, RED));

    // 添加图例
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()
        .context("draw legends")?;

    // TODO(xiangminli): 找到画第二个轴的方法

    // 保存图表
    root.present().context("present")?;

    Ok(())
}
