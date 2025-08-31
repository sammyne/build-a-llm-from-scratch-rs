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

    let c = df
        .clone()
        .lazy()
        .select([col("Label").value_counts(false, false, "count", false)])
        .collect()
        .context("count values")?;
    println!("{c}");

    Ok(())
}
