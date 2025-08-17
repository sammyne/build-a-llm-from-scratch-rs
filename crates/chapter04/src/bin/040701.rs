use std::collections::HashSet;

use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::Tensor;
use tiktoken::ext::Encoding;

type B = Autodiff<NdArray<f32>>;

fn main() {
    B::seed(123);

    let device = &<B as Backend>::Device::default();

    let start_context = "Hello, I am";

    let tokenizer = Encoding::gpt2();
    let encoded = {
        let allowed_specials = HashSet::default();
        tokenizer.encode(start_context, &allowed_specials)
    };
    let expected = [15496u32, 11, 314, 716];
    assert_eq!(&expected, encoded.as_slice(), "unexpected encoded start-context");
    println!("encoded: {encoded:?}");

    let encoded_tensor = Tensor::<B, 1, Int>::from_ints(encoded.as_slice(), &device).unsqueeze::<2>();
    println!("encoded-tensor.shape: {:?}", encoded_tensor.shape());
}
