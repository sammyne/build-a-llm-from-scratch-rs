use burn::prelude::Backend;
use burn::tensor::{Distribution, Int, Tensor};

use crate::x::TensorExt;

/// 多项式采样函数
/// 参数：`probs` - 2 维概率数组。
/// 返回值：采样到的对象下标（索引）
/// 假设 probs 的最后一维每个元素的和为 1。
pub fn multinomial<B: Backend>(probas: Tensor<B, 2>) -> Tensor<B, 2, Int> {
    // let ndim = *probas.dims().last().expect("get last dim");
    let [d0, _] = probas.dims();
    let device = &probas.device();

    // // 1. 计算累计概率。
    // let c = probas.matmul(Tensor::<B, 2>::ones([ndim, ndim], device).triu(0));
    let p = probas.cumsum();

    // // 2. 生成随机数
    let r = Tensor::<B, 1>::random([d0], Distribution::Uniform(0.0, 1.0), device);

    // 3. 搜索第一个满足 p>r 的下标
    p.search_sorted(r).unsqueeze()
}
