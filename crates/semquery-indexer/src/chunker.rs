use semquery_core::{ChunkCandidate, Chunker};
use tokenizers::Tokenizer;

pub struct SentenceSplitter {
  tokenizer: Tokenizer,
  chunk_size: usize,
  chunk_overlap: usize,
}

impl SentenceSplitter {
  pub fn new(tokenizer: Tokenizer, chunk_size: usize, chunk_overlap: usize) -> Self {
    Self {
      tokenizer,
      chunk_size,
      chunk_overlap,
    }
  }

  fn token_count(&self, text: &str) -> usize {
    self.tokenizer.encode(text, false).map(|enc| enc.get_ids().len()).unwrap_or(text.chars().count())
  }

  fn split_paragraphs(text: &str) -> Vec<(&str, usize)> {
    let mut result = Vec::new();
    let mut search_start = 0;
    loop {
      let remaining = &text[search_start..];
      match remaining.find("\n\n\n") {
        Some(rel_pos) => {
          let abs_start = search_start;
          let abs_end = search_start + rel_pos;
          let para = &text[abs_start..abs_end];
          if !para.trim().is_empty() {
            result.push((para, abs_start));
          }
          search_start = abs_end + 3;
        }
        None => {
          let para = &text[search_start..];
          if !para.trim().is_empty() {
            result.push((para, search_start));
          }
          break;
        }
      }
    }
    result
  }

  fn split_sentences(para: &str, para_offset: usize) -> Vec<(&str, usize)> {
    let mut sentences = Vec::new();
    let mut char_start = 0;

    for (byte_idx, ch) in para.char_indices() {
      if matches!(ch, '.' | '!' | '?' | '。' | '！' | '？') {
        let byte_end = byte_idx + ch.len_utf8();
        let slice = &para[char_start..byte_end];
        if !slice.trim().is_empty() {
          sentences.push((slice, para_offset + char_start));
        }
        char_start = byte_end;
      }
    }

    if char_start < para.len() {
      let slice = &para[char_start..];
      if !slice.trim().is_empty() {
        sentences.push((slice, para_offset + char_start));
      }
    }

    if sentences.is_empty() && !para.trim().is_empty() {
      vec![(para, para_offset)]
    } else {
      sentences
    }
  }

  fn split_words(sentence: &str, sentence_offset: usize) -> Vec<(&str, usize)> {
    let mut result = Vec::new();
    let mut search_start = 0;
    while let Some(rel) = sentence[search_start..].find(|c: char| !c.is_whitespace()) {
      let word_start = search_start + rel;
      let remaining = &sentence[word_start..];
      let word_end = remaining.find(|c: char| c.is_whitespace()).map(|p| word_start + p).unwrap_or(sentence.len());
      result.push((&sentence[word_start..word_end], sentence_offset + word_start));
      search_start = word_end;
    }
    result
  }

  fn split_chars(text: &str, text_offset: usize) -> Vec<(&str, usize)> {
    text
      .char_indices()
      .map(|(byte_idx, ch)| {
        let s = &text[byte_idx..byte_idx + ch.len_utf8()];
        (s, text_offset + byte_idx)
      })
      .collect()
  }
}

impl Chunker for SentenceSplitter {
  fn chunk(&self, text: &str) -> Vec<ChunkCandidate> {
    if text.trim().is_empty() {
      return Vec::new();
    }

    // Each unit: (text slice, byte offset in original text, token count)
    let mut units: Vec<(&str, usize, usize)> = Vec::new();
    for (para, para_offset) in Self::split_paragraphs(text) {
      for (sentence, sentence_offset) in Self::split_sentences(para, para_offset) {
        let st = self.token_count(sentence);
        if st <= self.chunk_size {
          units.push((sentence, sentence_offset, st));
        } else {
          for (word, word_offset) in Self::split_words(sentence, sentence_offset) {
            let wt = self.token_count(word);
            if wt <= self.chunk_size {
              units.push((word, word_offset, wt));
            } else {
              for (ch, ch_offset) in Self::split_chars(word, word_offset) {
                units.push((ch, ch_offset, 1));
              }
            }
          }
        }
      }
    }

    let mut chunks: Vec<ChunkCandidate> = Vec::new();
    let mut current: Vec<(&str, usize, usize)> = Vec::new();
    let mut current_tokens = 0usize;

    for &(unit, unit_offset, unit_tokens) in &units {
      if current_tokens + unit_tokens > self.chunk_size && !current.is_empty() {
        let start = current[0].1;
        let end = {
          let last = current.last().unwrap();
          last.1 + last.0.len()
        };
        let chunk_text = text[start..end].to_string();
        chunks.push(ChunkCandidate {
          text: chunk_text,
          byte_range: start..end,
        });

        let mut overlap_units: Vec<(&str, usize, usize)> = Vec::new();
        let mut overlap_tokens = 0usize;
        for &u in current.iter().rev() {
          if overlap_tokens + u.2 > self.chunk_overlap {
            break;
          }
          overlap_tokens += u.2;
          overlap_units.insert(0, u);
        }

        current = overlap_units;
        current_tokens = overlap_tokens;
      }

      current.push((unit, unit_offset, unit_tokens));
      current_tokens += unit_tokens;
    }

    if !current.is_empty() {
      let start = current[0].1;
      let end = {
        let last = current.last().unwrap();
        last.1 + last.0.len()
      };
      let chunk_text = text[start..end].to_string();
      chunks.push(ChunkCandidate {
        text: chunk_text,
        byte_range: start..end,
      });
    }

    chunks
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tokenizers::Tokenizer;
  use tokenizers::models::wordlevel::WordLevelBuilder;
  use tokenizers::pre_tokenizers::whitespace::Whitespace;

  fn test_tokenizer() -> Tokenizer {
    let mut vocab = std::collections::HashMap::new();
    vocab.insert("[UNK]".to_string(), 0u32);
    vocab.insert("[PAD]".to_string(), 1u32);
    for (i, c) in ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789").chars().enumerate() {
      vocab.insert(c.to_string(), (i + 2) as u32);
    }
    let model = WordLevelBuilder::new().vocab(vocab).unk_token("[UNK]".to_string()).build().unwrap();
    let mut tok = Tokenizer::new(model);
    tok.with_pre_tokenizer(Some(Whitespace));
    tok
  }

  #[test]
  fn test_short_text_single_chunk() {
    let tok = test_tokenizer();
    let splitter = SentenceSplitter::new(tok, 100, 10);
    let text = "Hello world. This is a test.";
    let chunks = splitter.chunk(text);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].byte_range.start, 0);
    assert_eq!(chunks[0].byte_range.end, text.len());
    assert_eq!(chunks[0].text, text);
  }

  #[test]
  fn test_multiple_sentences_multiple_chunks() {
    let tok = test_tokenizer();
    let splitter = SentenceSplitter::new(tok, 3, 1);
    let text = "Hello world. Foo bar. Baz qux.";
    let chunks = splitter.chunk(text);
    assert!(chunks.len() >= 2, "expected at least 2 chunks, got {}", chunks.len());
    for ch in &chunks {
      assert!(ch.byte_range.end > ch.byte_range.start);
      assert!(!ch.text.is_empty());
    }
  }

  #[test]
  fn test_byte_ranges_cover_text() {
    let tok = test_tokenizer();
    let splitter = SentenceSplitter::new(tok, 5, 2);
    let text = "First sentence here. Second one too. Third is short.";
    let chunks = splitter.chunk(text);
    assert!(!chunks.is_empty());
    assert_eq!(chunks[0].byte_range.start, 0);
    for w in chunks.windows(2) {
      assert!(
        w[0].byte_range.end >= w[1].byte_range.start,
        "chunk end {} < next start {}",
        w[0].byte_range.end,
        w[1].byte_range.start
      );
    }
    let last = chunks.last().unwrap();
    assert!(last.byte_range.end <= text.len());
  }

  #[test]
  fn test_chinese_punctuation_split() {
    let tok = test_tokenizer();
    let splitter = SentenceSplitter::new(tok, 100, 10);
    let text = "这是一个句子。这是另一个句子！还有第三个？";
    let chunks = splitter.chunk(text);
    assert_eq!(chunks.len(), 1, "with chunk_size=100 all 3 sentences fit in one chunk");
    assert_eq!(chunks[0].text, text);
    assert_eq!(chunks[0].byte_range, 0..text.len());
  }

  #[test]
  fn test_empty_text() {
    let tok = test_tokenizer();
    let splitter = SentenceSplitter::new(tok, 100, 10);
    let chunks = splitter.chunk("");
    assert!(chunks.is_empty());
  }

  #[test]
  fn test_paragraph_split() {
    let tok = test_tokenizer();
    let splitter = SentenceSplitter::new(tok, 100, 10);
    let text = "Paragraph one here.\n\n\nParagraph two here.";
    let chunks = splitter.chunk(text);
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].text.contains("Paragraph one"));
    assert!(chunks[0].text.contains("Paragraph two"));
  }

  #[test]
  fn test_overlap_between_chunks() {
    let tok = test_tokenizer();
    let splitter = SentenceSplitter::new(tok, 2, 1);
    let text = "alpha beta gamma delta epsilon zeta";
    let chunks = splitter.chunk(text);
    assert!(chunks.len() >= 2, "expected at least 2 chunks, got {}", chunks.len());
    if chunks.len() >= 2 {
      assert!(
        chunks[0].byte_range.end > chunks[1].byte_range.start,
        "expected overlap: first.end={} > second.start={}",
        chunks[0].byte_range.end,
        chunks[1].byte_range.start
      );
    }
  }

  #[test]
  fn test_byte_ranges_match_original_text() {
    let tok = test_tokenizer();
    let splitter = SentenceSplitter::new(tok, 5, 1);
    let text = "First sentence here. Second one too. Third is short.";
    let chunks = splitter.chunk(text);
    for ch in &chunks {
      let extracted = &text[ch.byte_range.start..ch.byte_range.end];
      assert_eq!(
        ch.text, *extracted,
        "byte_range text mismatch: chunk text != text[range]"
      );
    }
  }

  #[test]
  fn test_byte_ranges_with_multibyte_chars() {
    let tok = test_tokenizer();
    let splitter = SentenceSplitter::new(tok, 100, 10);
    let text = "你好世界。这是测试！";
    let chunks = splitter.chunk(text);
    assert_eq!(chunks.len(), 1);
    let extracted = &text[chunks[0].byte_range.start..chunks[0].byte_range.end];
    assert_eq!(chunks[0].text, extracted);
  }

  #[test]
  fn test_byte_ranges_with_paragraphs() {
    let tok = test_tokenizer();
    let splitter = SentenceSplitter::new(tok, 100, 10);
    let text = "First paragraph.\n\n\nSecond paragraph.";
    let chunks = splitter.chunk(text);
    assert_eq!(chunks.len(), 1);
    let extracted = &text[chunks[0].byte_range.start..chunks[0].byte_range.end];
    assert_eq!(chunks[0].text, extracted);
  }
}
