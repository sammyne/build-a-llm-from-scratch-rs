use anyhow::Context;
use burn::backend::NdArray;
use burn::nn::Relu;
use burn::prelude::Backend;
use burn::tensor::{DType, Int, Tensor};
use chapter04::Gelu;
use plotters::prelude::*;

type B = NdArray<f32>;

fn main() -> anyhow::Result<()> {
    let device = &<B as Backend>::Device::default();

    const OUT: &str = "gelu-vs-relu.svg";

    let root = SVGBackend::new(OUT, (700, 350)).into_drawing_area();
    root.fill(&WHITE).context("fill drawing area")?;

    let mut chart = ChartBuilder::on(&root)
        .caption("GELU vs ReLU", ("sans-serif", 30).into_font())
        .margin(5)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(-3f32..4f32, -0.5f32..3f32)?;

    chart
        .configure_mesh()
        .x_labels(8)
        .y_labels(4)
        .draw()
        .context("setup mesh")?;

    B::seed(123);
    let x = Tensor::<B, 1, Int>::arange_step(-300..300, 600 / 100, device).float() / 100.0;

    let y_gelu: Vec<f32> = Gelu
        .forward(x.clone().unsqueeze::<3>())
        .squeeze_dims::<1>(&[0, 1])
        .to_data()
        .convert_dtype(DType::F32)
        .into_vec()
        .map_err(|err| anyhow::anyhow!("y-gelu as f32 vec: {err:?}"))?;

    let y_relu: Vec<f32> = Relu
        .forward(x.clone())
        .to_data()
        .convert_dtype(DType::F32)
        .into_vec()
        .map_err(|err| anyhow::anyhow!("y-relu as f32 vec: {err:?}"))?;

    let x: Vec<f32> = x
        .to_data()
        .into_vec()
        .map_err(|err| anyhow::anyhow!("x as scalar vec: {err:?}"))?;

    chart
        .draw_series(LineSeries::new(x.iter().cloned().zip(y_gelu), BLUE.stroke_width(2)))
        .context("draw x-gelu(x) series")?
        .label("y = gelu(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(x.iter().cloned().zip(y_relu), RED.stroke_width(2)))
        .context("draw x-relu(x) series")?
        .label("y = relu(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .context("draw series labels")?;

    root.present().context("present")?;

    println!("图已绘制在 {OUT}");

    Ok(())
}
