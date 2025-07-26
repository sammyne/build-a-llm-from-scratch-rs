use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use chapter05::rand;

fn main() {
    type B = NdArray;

    let device = &<B as Backend>::Device::default();

    let p = Tensor::<B, 2>::from_floats([[0.1, 0.2, 0.3, 0.4], [0.1, 0.3, 0.3, 0.3]], device);

    let got = rand::multinomial(p).squeeze::<1>(0);

    println!("{got}");
}
