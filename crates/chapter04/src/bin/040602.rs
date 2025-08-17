use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use chapter04::GPT_124M;

type B = Autodiff<NdArray<f32>>;

fn main() {
    B::seed(123);

    let device = &<B as Backend>::Device::default();

    let model = GPT_124M.init::<B>(device);

    // 具体参数分布如下
    // 1. token-embedding: vocab-size * embed-dim = 50427 * 768 = 38727936
    // 2. position-embedding: context-length * embed-dim = 1024 * 768 = 786432
    // 3. dropout: 0
    // 4. 12 个 transformer-block: ()
    //  - layer-norm: 1536
    //     - eps: 0
    //     - scale: 768
    //     - shift: 768
    //  - MHA:
    //    - Q: embed-dim * embed-dim = 768 * 768 = 589824
    //    - K: embed-dim * embed-dim = 768 * 768 = 589824
    //    - V: embed-dim * embed-dim = 768 * 768 = 589824
    //    - biased-out-proj: (embed-dim + 1) * embed-dim = (768+1)*768 = 590592
    //    - dropout: 0
    //  - dropout: 0
    //  - layer-norm: 1536
    //     - eps: 0
    //     - scale: 768
    //     - shift: 768
    //  - dropout: 0
    //  - feed-forward:
    //    - biased-linear1: (embed-dim + 1) * (embed-dim * 4) = (768+1)*(768*4) = 2362368
    //    - gelu: 0
    //    - biased-linear2: (embed-dim * 4 + 1) * 768 = (768*4+1)*768 = 2360064
    //  - dropout: 0
    // 5. layer-norm: 1536
    //   - eps: 0
    //   - scale: 768
    //   - shift: 768
    // 6. out-head: embed-dim * vocab-size = 768 * 50427 = 38597376
    let total_params = model.num_params();
    assert_eq!(163_009_536, total_params, "unexpected #(parameters)");

    println!("Total number of parameters: {total_params}");
}
