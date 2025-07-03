use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::MseLoss;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use chapter04::Gelu;

type B = Autodiff<NdArray<f32>>;

fn main() -> anyhow::Result<()> {
    let device = <B as Backend>::Device::default();
    let layer_sizes = [3usize, 3, 3, 3, 3, 1];

    let sample_input = Tensor::<B, 2>::from_floats([[1.0, 0.0, -1.0]], &device).unsqueeze::<3>();

    B::seed(123);

    println!("model without shortcut");
    let model_without_shortcut = ExampleDeepNeuralNetwork::<B>::new(&layer_sizes, false);
    print_gradients(model_without_shortcut, sample_input.clone());

    B::seed(123);
    println!("\nmodel with shortcut");
    let model_with_shortcut = ExampleDeepNeuralNetwork::<B>::new(&layer_sizes, true);
    print_gradients(model_with_shortcut, sample_input.clone());

    Ok(())
}

#[derive(Debug, Module)]
struct ExampleDeepNeuralNetwork<B: Backend> {
    use_shortcut: bool,
    layers: Vec<(Linear<B>, Gelu)>,
}

impl<B: Backend> ExampleDeepNeuralNetwork<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = x;
        for (l, g) in &self.layers {
            let layer_output = l.forward(x.clone());
            let layer_output = g.forward(layer_output);

            if self.use_shortcut && x.shape() == layer_output.shape() {
                x = x.clone() + layer_output;
            } else {
                x = layer_output;
            }
        }

        x
    }

    pub fn new(layer_sizes: &[usize], use_shortcut: bool) -> Self {
        let device = B::Device::default();

        let layers = vec![
            (LinearConfig::new(layer_sizes[0], layer_sizes[1]).init(&device), Gelu),
            (LinearConfig::new(layer_sizes[1], layer_sizes[2]).init(&device), Gelu),
            (LinearConfig::new(layer_sizes[2], layer_sizes[3]).init(&device), Gelu),
            (LinearConfig::new(layer_sizes[3], layer_sizes[4]).init(&device), Gelu),
            (LinearConfig::new(layer_sizes[4], layer_sizes[5]).init(&device), Gelu),
        ];

        Self { use_shortcut, layers }
    }
}

fn print_gradients(model: ExampleDeepNeuralNetwork<B>, x: Tensor<B, 3>) {
    let device = x.device();

    let output = model.forward(x);

    let target = Tensor::<B, 2>::from_floats([[0.0]], &device).unsqueeze::<3>();

    let loss = MseLoss::new();
    let loss = loss.forward_no_reduction(output, target);

    let gradients = loss.backward();
    for (i, l) in model.layers.iter().map(|v| &v.0).enumerate() {
        match l.weight.grad(&gradients) {
            Some(v) => {
                println!("{i}-th layer has gradient mean of {}", v.abs().mean().into_scalar());
            }
            None => println!("miss gradient for {i}-th layer"),
        }
    }


}
