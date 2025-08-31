use std::path::Path;

use anyhow::Context;
use polars::prelude::*;

fn main() -> anyhow::Result<()> {
    let path = Path::new("sms_spam_collection/SMSSpamCollection.tsv");
    if !path.exists() {
        anyhow::bail!("{path:?} not found");
    }

    let schema: Schema = [("Label".into(), DataType::String), ("Text".into(), DataType::String)]
        .into_iter()
        .collect();

    let df = CsvReadOptions::default()
        .with_has_header(false)
        .with_schema(Some(Arc::new(schema)))
        .with_infer_schema_length(None)
        .map_parse_options(|v| v.with_separator(b'\t').with_quote_char(None))
        .try_into_reader_with_file_path(Some(path.into()))
        .context("new csv reader")?
        .finish()
        .context("finish reading csv")?;

    // 这里能看到 5574 行记录
    // 原书用 pandas 只能读取到 5572 行，具体原因参见这里 https://github.com/rasbt/LLMs-from-scratch/discussions/757。
    println!("{df}");

    Ok(())
}
