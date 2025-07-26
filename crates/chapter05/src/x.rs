use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};

pub trait TensorExt<B: Backend> {
    fn cumsum(self) -> Self;

    fn search_sorted(self, values: Tensor<B, 1>) -> Tensor<B, 1, Int>;
}

impl<B: Backend> TensorExt<B> for Tensor<B, 2> {
    fn cumsum(self) -> Self {
        // 实现方式 1
        // let device = &self.device();
        // let [_, d1] = self.dims();
        // let v = Tensor::<B, 2>::ones([d1, d1], device).triu(0);
        // self.matmul(v)

        // 实现方式 2：解决大 d1 导致 #1 OOM 的问题
        let data: Vec<_> = self
            .iter_dim(0)
            .map(|v| v.squeeze::<1>(0).cumsum())
            .collect();
        Tensor::stack::<2>(data, 0)
    }

    fn search_sorted(self, values: Tensor<B, 1>) -> Tensor<B, 1, Int> {
        // 实现方式 1
        let device = &self.device();

        // let [d0, d1] = self.dims();
        assert_eq!(self.dims()[0], values.dims()[0], "bad values len");

        // let values = values.unsqueeze::<2>().transpose().expand([d0, d1]);

        // let discarded = self.lower_equal(values);

        // let indices = Tensor::<B, 1, Int>::arange(0..(d1 as i64), device).unsqueeze::<2>();

        // let indices = indices.expand([d0, d1]);

        // let indices = indices.mask_fill(discarded, d1 as i64);

        // let out = indices.min_dim(1).transpose().squeeze::<1>(0);

        // 实现方式 2：解决大 [d0,d1] 导致方式 1 OOM 的问题
        let tensors: Vec<_> = self
            .iter_dim(0)
            .map(|v| v.squeeze::<1>(0))
            .zip(values.iter_dim(0))
            .map(|(v, t)| v.search_sorted(t).into_scalar())
            .collect();
        let out = Tensor::from_ints(tensors.as_slice(), device);

        out
    }
}

impl<B: Backend> TensorExt<B> for Tensor<B, 1> {
    fn cumsum(self) -> Self {
        let device = &self.device();

        // println!("dtype {:?}", self.dtype());
        let mut data = self.to_data().to_vec::<f32>().expect("unwrap as Vec<f32>");
        let mut s = 0.0f32;
        for v in data.iter_mut() {
            let vv = *v;
            *v += s;
            s += vv;
        }

        Self::from_floats(data.as_slice(), device)
    }

    fn search_sorted(self, values: Tensor<B, 1>) -> Tensor<B, 1, Int> {
        let device = &self.device();

        let data = self.to_data().to_vec::<f32>().expect("unwrap self as Vec<f32>");
        let v = values.to_data().to_vec::<f32>().expect("unwrap value as f32")[0];
        let i = search_sorted(&data, v);

        Tensor::from_ints([i], device)
    }
}

// 类似 torch.searchsorted(..,right=True,..)
pub fn search_sorted(v: &[f32], t: f32) -> usize {
    let mut left = 0;
    let mut right = v.len();

    while left < right {
        let mid = left + (right - left) / 2;
        // 安全访问：mid ∈ [0, len)
        unsafe {
            match v.get_unchecked(mid).partial_cmp(&t).expect("NaN not allowed") {
                // 关键调整：相等时继续向右搜索
                std::cmp::Ordering::Equal | std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Greater => right = mid,
            }
        }
    }

    right
}
