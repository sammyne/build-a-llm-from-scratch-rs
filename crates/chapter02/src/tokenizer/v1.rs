use std::collections::HashMap;

use crate::strings;

pub struct TokenizerV1 {
    ids: HashMap<String, usize>,
    strs: HashMap<usize, String>,
}

impl TokenizerV1 {
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|i| &self.strs[i]).fold(String::new(), |acc, s| {
            if acc.is_empty() { s.to_owned() } else { concat(acc, s) }
        })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        strings::split(text)
            .into_iter()
            .filter(|v| !v.trim().is_empty())
            .map(|v| self.ids[v])
            .collect()
    }

    pub fn new<T: IntoIterator<Item = String>>(vocab: T) -> Self {
        let ids: HashMap<String, usize> = vocab.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        let strs = ids.iter().map(|(v, i)| (*i, v.to_owned())).collect();

        Self { ids, strs }
    }
}

fn concat(a: String, token: &str) -> String {
    if a.is_empty() {
        return token.to_owned();
    }

    const P: &str = r#",.?!"()\\'"#;
    if P.contains(token) { a + token } else { a + " " + token }
}
