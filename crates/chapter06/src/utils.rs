use std::sync::Arc;

use anyhow::Context as _;
use burn::module::{ModuleMapper, ParamId, Parameter as _};
use burn::prelude::Backend;
use burn::tensor::{Bool, Int, Tensor};
use polars::frame::DataFrame;
use polars::io::SerReader as _;
use polars::prelude::{CsvReadOptions, DataType, Schema};

pub struct RequireGradMapper;

impl<B: Backend> ModuleMapper<B> for RequireGradMapper {
    fn map_float<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        tensor.set_require_grad(true)
    }

    fn map_int<const D: usize>(&mut self, _id: burn::module::ParamId, tensor: Tensor<B, D, Int>) -> Tensor<B, D, Int> {
        tensor.set_require_grad(true)
    }

    fn map_bool<const D: usize>(&mut self, _id: ParamId, tensor: Tensor<B, D, Bool>) -> Tensor<B, D, Bool> {
        tensor.set_require_grad(true)
    }
}

pub fn load_csv(path: &str) -> anyhow::Result<DataFrame> {
    let schema: Schema = [("Label".into(), DataType::UInt32), ("Text".into(), DataType::String)]
        .into_iter()
        .collect();

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_schema(Some(Arc::new(schema)))
        .try_into_reader_with_file_path(Some(path.into()))
        .context("new csv reader")?
        .finish()
        .context("finish reading csv")?;

    Ok(df)
}
