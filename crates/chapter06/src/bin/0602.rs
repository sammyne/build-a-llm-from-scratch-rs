use std::{fs::File, path::Path};

use anyhow::Context;
use polars::prelude::*;

fn main() -> anyhow::Result<()> {
    let path = Path::new("sms_spam_collection/SMSSpamCollection.tsv");
    if !path.exists() {
        anyhow::bail!("{:?} not found", path);
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

    println!("{df}");

    let c = df
        .clone()
        .lazy()
        .select([col("Label").value_counts(false, false, "count", false)])
        .collect()
        .context("count values")?;
    println!("{}", c);

    let balanced_df = create_balanced_dataset(df.clone()).context("create balanced dataset")?;
    println!("{}", balanced_df.clone().count().collect().expect("count balanced df"));

    let balanced_df = balanced_df.with_column(when(col("Label").eq(lit("spam"))).then(1).otherwise(0).alias("Label"));
    println!("{}", balanced_df.clone().collect().expect("count balanced df"));

    let (train_df, validation_df, test_df) = random_split(balanced_df, 0.7, 0.1).context("split data")?;

    to_csv(train_df, "train.csv").context("save train df")?;
    to_csv(validation_df, "validation.csv").context("save validation df")?;
    to_csv(test_df, "test.csv").context("save test df")?;

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

// Listing 6.3. Splitting the dataset
fn random_split(
    df: LazyFrame,
    train_frac: f32,
    validation_frac: f32,
) -> anyhow::Result<(DataFrame, DataFrame, DataFrame)> {
    let df = df.collect().context("collect df")?;

    let f = [1.0].into_iter().collect::<Series>();
    let df = df.sample_frac(&f, false, true, 456.into()).context("sample")?;
    // println!("hello --- {df}");

    let train_len = (df.height() as f32 * train_frac) as usize;
    let validation_len = (df.height() as f32 * validation_frac) as usize;
    let test_len = df.height() - train_len - validation_len;

    let train = df.slice(0, train_len);
    let validation = df.slice(train_len as i64, validation_len);
    let test = df.slice((train_len + validation_len) as i64, test_len);

    Ok((train, validation, test))
}

fn to_csv(mut df: DataFrame, path: &str) -> anyhow::Result<()> {
    let mut out = File::create(path).context("create file")?;

    CsvWriter::new(&mut out)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df)
        .context("write")
}
