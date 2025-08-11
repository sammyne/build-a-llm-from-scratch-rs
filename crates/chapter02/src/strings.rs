use regex::Regex;

pub fn concat(a: String, token: &str) -> String {
    if a.is_empty() {
        return token.to_owned();
    }

    const P: &str = r#",.?!"()\\'"#;
    if P.contains(token) { a + token } else { a + " " + token }
}

pub fn split(s: &str, p: Option<Regex>) -> Vec<&str> {
    let r = match p {
        None => Regex::new(r#"([,.:;?_!"()'\\]|--|\s)"#).expect("build default regex"),
        Some(r) => r,
    };

    let mut out = vec![];
    let mut last = 0;
    for m in r.find_iter(s) {
        // 添加非分隔符部分
        if m.start() >= last {
            out.push(&s[last..m.start()]);
        }
        // 添加分隔符本身
        out.push(m.as_str());
        last = m.end();
    }
    // 添加剩余部分
    if last < s.len() {
        out.push(&s[last..]);
    }

    out
}
