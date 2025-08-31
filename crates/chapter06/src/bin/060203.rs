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

    let balanced_df = create_balanced_dataset(df.clone()).context("create balanced dataset")?;

    let c = balanced_df
        .select([col("Label").value_counts(false, false, "count", false)])
        .collect()
        .context("count values")?;
    println!("{c}");

    Ok(())
}

// Listing 6.2. Creating a balanced dataset
fn create_balanced_dataset(df: DataFrame) -> anyhow::Result<LazyFrame> {
    let spam_subset = df.clone().lazy().filter(col("Label").eq(lit("spam")));

    let num_spam: usize = spam_subset
        .clone()
        .drop(["Text"])
        .count()
        .collect()
        .context("collect #(spam)")?
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("miss value"))?[0]
        .try_extract()
        .context("get #(spam) as usize")?;

    let ham_subset = df
        .clone()
        .lazy()
        .filter(col("Label").eq(lit("ham")))
        .collect()
        .context("build full hams")?
        .sample_n_literal(num_spam, false, true, 123.into())
        .context("sample")?;

    concat([ham_subset.lazy(), spam_subset], Default::default()).context("concat hams and spams")
}
