use burn::prelude::*;
use burn::tensor::activation;

use crate::GptModel;

pub fn generate_text_simple<B: Backend>(
    model: &GptModel<B>,
    mut idx: Tensor<B, 2, Int>,
    max_new_tokens: usize,
    context_size: usize,
) -> Tensor<B, 2, Int> {
    let context_size = context_size as i32;
    for _ in 0..max_new_tokens {
        let idx_cond = idx.clone().slice(s![.., -context_size..]);
        let logits = model.forward(idx_cond);

        let logits = logits.slice(s![.., -1, ..]).squeeze(1);

        let dim = logits.dims().len() - 1;

        // softmax 是多余的。添加只是为了将输出转化为直观的概率。
        let probas = activation::softmax(logits, dim);
        let idx_next = probas.argmax(dim);
        idx = Tensor::cat(vec![idx, idx_next], 1);
    }

    idx
}
