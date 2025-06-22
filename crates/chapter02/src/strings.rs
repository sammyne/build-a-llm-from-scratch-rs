use regex::Regex;

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
