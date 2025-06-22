pub fn load() -> std::io::Result<String> {
    std::fs::read_to_string("static/the-verdict.txt")
}

pub fn load_and_canonicalize<T: FromIterator<String>>() -> std::io::Result<T> {
    let text = load()?;
    let out = crate::strings::split(&text)
        .into_iter()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
        .collect();
    Ok(out)
}
