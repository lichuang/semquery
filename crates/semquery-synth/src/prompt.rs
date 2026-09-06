use semquery_core::SearchHit;

pub fn build_ask_prompt(query: &str, hits: &[SearchHit]) -> String {
  let mut prompt = String::from("Context information is below.\n---------------------\n");

  for (i, hit) in hits.iter().enumerate() {
    let marker = i + 1;
    let chunk = &hit.chunk;
    prompt.push_str(&format!(
      "[{}] {} (bytes {}-{}):\n{}\n\n",
      marker,
      hit.file_path.display(),
      chunk.byte_range.start,
      chunk.byte_range.end,
      chunk.text
    ));
  }

  prompt.push_str("---------------------\n");
  prompt.push_str("Given the context information and not prior knowledge, answer the query.\n");
  prompt.push_str("Cite sources using [1], [2], etc.\n");
  prompt.push_str(&format!("Query: {query}\n"));
  prompt.push_str("Answer:");
  prompt
}

#[cfg(test)]
mod tests {
  use super::*;
  use semquery_core::{Chunk, ScoreExplain};

  fn make_hit(id: &str, doc_id: &str, text: &str, start: usize, end: usize) -> SearchHit {
    SearchHit {
      chunk: Chunk {
        id: id.into(),
        doc_id: doc_id.into(),
        text: text.into(),
        byte_range: start..end,
      },
      file_path: doc_id.into(),
      score: 0.1,
      explain: ScoreExplain::default(),
    }
  }

  #[test]
  fn test_build_ask_prompt_format() {
    let hits = vec![
      make_hit("c1", "docs/a.txt", "hello world", 0, 11),
      make_hit("c2", "docs/b.txt", "foo bar", 20, 27),
    ];
    let prompt = build_ask_prompt("What is a.txt about?", &hits);

    assert!(prompt.starts_with("Context information is below."));
    assert!(prompt.contains("[1] docs/a.txt (bytes 0-11):\nhello world\n"));
    assert!(prompt.contains("[2] docs/b.txt (bytes 20-27):\nfoo bar\n"));
    assert!(prompt.contains("Query: What is a.txt about?"));
    assert!(prompt.ends_with("Answer:"));
  }

  #[test]
  fn test_build_ask_prompt_empty_hits() {
    let prompt = build_ask_prompt("test", &[]);
    assert!(prompt.contains("Query: test"));
    assert!(prompt.ends_with("Answer:"));
  }
}
