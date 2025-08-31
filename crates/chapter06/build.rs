use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

use anyhow::Context;
use zip::ZipArchive;

// Listing 6.1 Downloading and unzipping the dataset
fn main() -> anyhow::Result<()> {
    // 备选 URL：https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip
    const URL: &str = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip";
    const ZIP_PATH: &str = "sms_spam_collection.zip";
    const EXTRACTED_PATH: &str = "sms_spam_collection";
    let data_file_path = PathBuf::from(EXTRACTED_PATH.to_owned()).join("SMSSpamCollection.tsv");

    if data_file_path.exists() {
        println!("{data_file_path:?} already exists. Skipping download and extraction.");
        return Ok(());
    }

    // 下载文件
    if !fs::exists(ZIP_PATH).with_context(|| format!("check existence of {ZIP_PATH}"))? {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .open(ZIP_PATH)
            .with_context(|| format!("open {ZIP_PATH}"))?;
        let _ = reqwest::blocking::get(URL)
            .context("http get")?
            .copy_to(&mut f)
            .with_context(|| format!("save http get response to {ZIP_PATH}"))?;
    }

    // 解压文件
    fs::remove_dir_all(EXTRACTED_PATH).with_context(|| format!("rm -rf {EXTRACTED_PATH}"))?;
    unzip(ZIP_PATH, EXTRACTED_PATH).context("unzip")?;

    // 重命名
    fs::rename(format!("{EXTRACTED_PATH}/SMSSpamCollection"), &data_file_path).context("rename")?;
    println!("File downloaded and saved as {data_file_path:?}");

    Ok(())
}

fn unzip<O: AsRef<Path>>(zip_path: &str, out_path: O) -> anyhow::Result<()> {
    let out_path = out_path.as_ref();
    if !fs::exists(out_path).context("check out-path existence")? {
        fs::create_dir(out_path).context("mkdir out-path")?;
    }

    let f = File::open(zip_path).context("open zip file")?;
    let mut z = ZipArchive::new(f).context("create zip archive")?;

    for i in 0..z.len() {
        let mut file = z.by_index(i).with_context(|| format!("get {i}-th file"))?;

        let path = out_path.join(Path::new(file.name()));
        {
            let p = path.parent().with_context(|| format!("get parent dir for {path:?}"))?;
            fs::create_dir_all(p).with_context(|| format!("mkdir parent dir {p:?} for {path:?}"))?;
        }

        let mut o = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .with_context(|| format!("open {path:?}"))?;

        std::io::copy(&mut file, &mut o).with_context(|| format!("copy {path:?} to {out_path:?}"))?;
    }

    Ok(())
}
