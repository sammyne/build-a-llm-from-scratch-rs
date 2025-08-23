use std::collections::HashMap;

use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::tensor::{ElementConversion, Tensor, activation};
use chapter05::rand;

type B = NdArray;

type D = <B as Backend>::Device;

fn main() {
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

    print_sampled_tokens(probas, &inverse_vocab);
}

fn print_sampled_tokens<B: Backend>(probas: Tensor<B, 1>, inverse_vocab: &HashMap<i64, &str>) {
    B::seed(123);

    let d = probas.dims()[0];

    let sample = (1..1000)
        .map(|_| {
            rand::multinomial(probas.clone().unsqueeze::<2>())
                .squeeze::<1>(0)
                .into_scalar()
        })
        .map(|v| v.elem::<i64>());

    let mut sampled_ids = vec![0; d];
    for i in sample {
        sampled_ids[i as usize] += 1;
    }

    for (i, freq) in sampled_ids.into_iter().enumerate() {
        let i = i as i64;
        println!("{} x {}", freq, inverse_vocab[&i]);
    }
}
