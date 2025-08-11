use std::collections::HashMap;

use crate::strings;
use crate::tokenizer::TOKEN_UNKNOWN;

pub struct SimpleTokenizerV2 {
    ids: HashMap<String, usize>,
    strs: HashMap<usize, String>,
}

impl SimpleTokenizerV2 {
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|i| &self.strs[i]).fold(String::new(), |acc, s| {
            if acc.is_empty() {
                s.to_owned()
            } else {
                strings::concat(acc, s)
            }
        })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let unknown = self.ids[TOKEN_UNKNOWN];

        strings::split(text, None)
            .into_iter()
            .filter(|v| !v.trim().is_empty())
            .map(|v| *self.ids.get(v).unwrap_or(&unknown))
            .collect()
    }

    pub fn new<T: IntoIterator<Item = String>>(vocab: T) -> Self {
        let ids: HashMap<String, usize> = vocab.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
        let strs = ids.iter().map(|(v, i)| (*i, v.to_owned())).collect();

        Self { ids, strs }
    }
}
