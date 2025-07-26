use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::tensor::{Float, Int, Tensor};
use chapter05::x::TensorExt;

type B = NdArray;

#[test]
fn cumsum_1d() {
    type T = Tensor<B, 1>;

    let device = &<B as Backend>::Device::default();

    struct Case {
        t: T,
        expect: T,
    }

    let test_vector = vec![
        Case {
            t: T::ones([3], device),
            expect: T::from_floats([1, 2, 3], device),
        },
        Case {
            t: T::from_floats([1., 2., 3.], device),
            expect: T::from_floats([1., 3., 6.], device),
        },
    ];

    for c in test_vector {
        let Case { t, expect } = c;

        let got = t.clone().cumsum();

        let matched = expect.clone().equal(got.clone()).all().into_scalar();
        assert!(matched, "given {t:?}, expect {expect:?}, got {got:?}");
    }
}

#[test]
fn cumsum_2d() {
    type T = Tensor<B, 2>;

    let device = &<B as Backend>::Device::default();

    struct Case {
        t: T,
        expect: T,
    }

    let test_vector = vec![
        Case {
            t: T::ones([1, 3], device),
            expect: T::from_floats([[1, 2, 3]], device),
        },
        Case {
            t: T::from_floats([[1., 2., 3.], [4., 5., 6.]], device),
            expect: T::from_floats([[1., 3., 6.], [4., 9., 15.]], device),
        },
    ];

    for c in test_vector {
        let Case { t, expect } = c;

        let got = t.clone().cumsum();

        let matched = expect.clone().equal(got.clone()).all().into_scalar();
        assert!(matched, "given {t:?}, expect {expect:?}, got {got:?}");
    }
}

#[test]
fn search_sorted_1d() {
    type Tensor1D<T = Float> = Tensor<B, 1, T>;

    let device = &<B as Backend>::Device::default();

    struct Case {
        t: Tensor1D,
        values: Tensor1D,
        expect: Tensor1D<Int>,
    }

    let test_vector = vec![
        Case {
            t: (Tensor::arange(1..11, device).float() / 10.0),
            values: Tensor1D::from_floats([0.3], device),
            expect: Tensor1D::from_ints([3], device),
        },
        Case {
            t: (Tensor::arange(1..11, device).float() / 10.0),
            values: Tensor1D::from_floats([0.99], device),
            expect: Tensor1D::from_ints([9], device),
        },
        Case {
            t: (Tensor::arange(1..11, device).float() / 10.0),
            values: Tensor1D::from_floats([0.01], device),
            expect: Tensor1D::from_ints([0], device),
        },
        Case {
            t: (Tensor::arange(1..11, device).float() / 10.0),
            values: Tensor1D::from_floats([0.], device),
            expect: Tensor1D::from_ints([0], device),
        },
        Case {
            t: Tensor::from_floats([0.1, 0.2, 0.2, 0.4, 0.8, 0.8, 1.0], device),
            values: Tensor1D::from_floats([0.2], device),
            expect: Tensor1D::from_ints([3], device),
        },
    ];

    for Case { t, values, expect } in test_vector {
        let got = t.clone().search_sorted(values.clone());

        let matched = expect.clone().equal(got.clone()).all().into_scalar();
        assert!(matched, "given\n{t}\nand\n{values}\nexpect\n{expect}\ngot\n{got}");
    }
}

#[test]
fn search_sorted_2d() {
    type T = Tensor<B, 2>;
    type Tensor1D<T = Float> = Tensor<B, 1, T>;

    let device = &<B as Backend>::Device::default();

    struct Case {
        t: T,
        values: Tensor1D,
        expect: Tensor1D<Int>,
    }

    let test_vector = vec![
        Case {
            t: (Tensor::arange(1..11, device).float() / 10.0).expand([2, 10]),
            values: Tensor1D::from_floats([0.3, 0.4], device),
            expect: Tensor1D::from_ints([3, 4], device),
        },
        Case {
            t: (Tensor::arange(1..11, device).float() / 10.0).reshape([1, 10]),
            values: Tensor1D::from_floats([0.99], device),
            expect: Tensor1D::from_ints([9], device),
        },
        Case {
            t: (Tensor::arange(1..11, device).float() / 10.0).reshape([1, 10]),
            values: Tensor1D::from_floats([0.01], device),
            expect: Tensor1D::from_ints([0], device),
        },
        Case {
            t: (Tensor::arange(1..11, device).float() / 10.0).reshape([1, 10]),
            values: Tensor1D::from_floats([0.], device),
            expect: Tensor1D::from_ints([0], device),
        },
        Case {
            t: Tensor::from_floats([[0.1, 0.2, 0.2, 0.4, 0.8, 0.8, 1.0]], device),
            values: Tensor1D::from_floats([0.2], device),
            expect: Tensor1D::from_ints([3], device),
        },
    ];

    for Case { t, values, expect } in test_vector {
        let got = t.clone().search_sorted(values.clone());

        let matched = expect.clone().equal(got.clone()).all().into_scalar();
        assert!(matched, "given\n{t}\nand\n{values}\nexpect\n{expect}\ngot\n{got}");
    }
}
