use std::sync::Arc;

use anyhow::Context as _;
use polars::frame::DataFrame;
use polars::io::SerReader as _;
use polars::prelude::{CsvReadOptions, DataType, Schema};

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
