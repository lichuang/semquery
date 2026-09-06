use std::path::Path;

use semquery_core::{FileReader, ParseError, Result};

pub struct TextFileReader {
  extensions: Vec<&'static str>,
}

impl Default for TextFileReader {
  fn default() -> Self {
    Self::new()
  }
}

impl TextFileReader {
  pub fn new() -> Self {
    Self::with_extensions(&["txt", "md", "tex"])
  }

  pub fn with_extensions(extensions: &[&'static str]) -> Self {
    Self {
      extensions: extensions.to_vec(),
    }
  }
}

impl FileReader for TextFileReader {
  fn extensions(&self) -> &[&str] {
    &self.extensions
  }

  fn read(&self, path: &Path) -> Result<Option<semquery_core::DocumentSource>> {
    match std::fs::read_to_string(path) {
      Ok(content) => {
        if content.is_empty() {
          Ok(None)
        } else {
          Ok(Some(semquery_core::DocumentSource {
            path: path.to_path_buf(),
            content,
          }))
        }
      }
      Err(e) if e.kind() == std::io::ErrorKind::InvalidData => Ok(None),
      Err(e) => Err(
        ParseError::Io {
          path: path.display().to_string(),
          source: e,
        }
        .into(),
      ),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::fs;
  use tempfile::TempDir;

  #[test]
  fn test_txt_reader_reads_utf8_file() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("a.txt");
    fs::write(&path, "hello world").unwrap();

    let reader = TextFileReader::new();
    let doc = reader.read(&path).unwrap().unwrap();
    assert_eq!(doc.content, "hello world");
  }

  #[test]
  fn test_txt_reader_skips_empty_file() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("empty.txt");
    fs::write(&path, "").unwrap();

    let reader = TextFileReader::new();
    assert!(reader.read(&path).unwrap().is_none());
  }

  #[test]
  fn test_txt_reader_skips_non_utf8() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("bad.txt");
    fs::write(&path, [0xFF, 0xFE, 0xFD]).unwrap();

    let reader = TextFileReader::new();
    assert!(reader.read(&path).unwrap().is_none());
  }
}
