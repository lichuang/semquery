use jieba_rs::Jieba;
use semquery_core::WordSegmenter;

static JIEBA: std::sync::OnceLock<Jieba> = std::sync::OnceLock::new();

fn jieba() -> &'static Jieba {
  JIEBA.get_or_init(Jieba::new)
}

pub struct JiebaSegmenter;

impl WordSegmenter for JiebaSegmenter {
  fn segment(&self, text: &str) -> String {
    jieba().cut(text, false).join(" ")
  }
}

impl Default for JiebaSegmenter {
  fn default() -> Self {
    Self
  }
}

pub fn jieba_tokenize(text: &str) -> String {
  JiebaSegmenter.segment(text)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_jieba_tokenize_basic() {
    let result = jieba_tokenize("分布式共识算法");
    let tokens: Vec<&str> = result.split_whitespace().collect();
    assert!(!tokens.is_empty());
    assert!(tokens.contains(&"共识"));
    assert!(tokens.contains(&"算法"));
  }

  #[test]
  fn test_jieba_tokenize_english() {
    let result = jieba_tokenize("hello world");
    let tokens: Vec<&str> = result.split_whitespace().collect();
    assert!(tokens.contains(&"hello"));
    assert!(tokens.contains(&"world"));
  }

  #[test]
  fn test_jieba_tokenize_mixed() {
    let result = jieba_tokenize("Raft是一种共识算法");
    assert!(result.contains("共识"));
    assert!(result.contains("算法"));
  }

  #[test]
  fn test_jieba_segmenter_via_trait() {
    let seg: &dyn WordSegmenter = &JiebaSegmenter;
    let result = seg.segment("共识算法");
    assert!(result.contains("共识"));
  }
}
