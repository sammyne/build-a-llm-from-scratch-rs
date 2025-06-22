use regex::Regex;

pub fn concat(a: String, token: &str) -> String {
    if a.is_empty() {
        return token.to_owned();
    }

    const P: &str = r#",.?!"()\\'"#;
    if P.contains(token) { a + token } else { a + " " + token }
}

pub fn split(text: &str) -> Vec<&str> {
    let re = Regex::new(r#"([,.:;?_!"()'\\]|--|\s)"#).unwrap();
    let mut result = Vec::new();
    let mut last = 0;

    for cap in re.captures_iter(text) {
        let m = cap.get(0).unwrap();
        if m.start() > last {
            result.push(&text[last..m.start()]);
        }
        result.push(m.as_str());
        last = m.end();
    }

    if last < text.len() {
        result.push(&text[last..]);
    }

    result
}
