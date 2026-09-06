use std::io::{Cursor, Read};
use std::path::Path;

use quick_xml::Reader;
use quick_xml::events::Event;
use semquery_core::{FileReader, ParseError, Result};

pub struct DocxReader;

impl Default for DocxReader {
  fn default() -> Self {
    Self::new()
  }
}

impl DocxReader {
  pub fn new() -> Self {
    Self
  }
}

impl FileReader for DocxReader {
  fn extensions(&self) -> &[&str] {
    &["docx"]
  }

  fn read(&self, path: &Path) -> Result<Option<semquery_core::DocumentSource>> {
    let bytes = std::fs::read(path).map_err(|e| ParseError::Io {
      path: path.display().to_string(),
      source: e,
    })?;
    let cursor = Cursor::new(&bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| ParseError::ZipFailed {
      path: path.display().to_string(),
      message: e.to_string(),
    })?;

    let mut xml = String::new();
    {
      let mut entry = archive.by_name("word/document.xml").map_err(|e| ParseError::ZipEntryMissing {
        path: path.display().to_string(),
        entry: "word/document.xml".into(),
        message: e.to_string(),
      })?;
      entry.read_to_string(&mut xml).map_err(|e| ParseError::ZipEntryMissing {
        path: path.display().to_string(),
        entry: "word/document.xml".into(),
        message: e.to_string(),
      })?;
    }

    let text = extract_text(&xml)?;
    if text.trim().is_empty() {
      Ok(None)
    } else {
      Ok(Some(semquery_core::DocumentSource {
        path: path.to_path_buf(),
        content: text,
      }))
    }
  }
}

fn extract_text(xml: &str) -> Result<String> {
  let mut reader = Reader::from_str(xml);

  let mut buf = Vec::new();
  let mut text = String::new();
  let mut in_text_node = false;

  loop {
    match reader.read_event_into(&mut buf) {
      Ok(Event::Start(e)) => match e.name().as_ref() {
        b"w:t" => in_text_node = true,
        b"w:tab" => text.push('\t'),
        b"w:br" => text.push('\n'),
        _ => {}
      },
      Ok(Event::Text(e)) if in_text_node => {
        let chunk = e.unescape().map_err(|err| ParseError::XmlParseFailed {
          message: format!("unescape docx text: {err}"),
        })?;
        text.push_str(&chunk);
      }
      Ok(Event::End(e)) if e.name().as_ref() == b"w:t" => {
        in_text_node = false;
      }
      Ok(Event::Eof) => break,
      Err(e) => return Err(ParseError::XmlParseFailed { message: e.to_string() }.into()),
      _ => {}
    }
    buf.clear();
  }

  Ok(text)
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::fs;
  use std::io::Write;
  use tempfile::TempDir;

  fn docx_bytes_with_text(text: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    {
      let mut zip = zip::ZipWriter::new(Cursor::new(&mut buf));
      let options = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
      let xml = format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:r>
        <w:t>{}</w:t>
      </w:r>
    </w:p>
  </w:body>
</w:document>"#,
        quick_xml::escape::escape(text)
      );
      zip.start_file("word/document.xml", options).unwrap();
      zip.write_all(xml.as_bytes()).unwrap();
      zip.finish().unwrap();
    }
    buf
  }

  #[test]
  fn test_docx_reader_extracts_text() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("sample.docx");
    fs::write(&path, docx_bytes_with_text("Hello DOCX")).unwrap();

    let reader = DocxReader::new();
    let doc = reader.read(&path).unwrap().unwrap();
    assert!(doc.content.contains("Hello DOCX"));
  }

  #[test]
  fn test_docx_reader_skips_empty_text() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("empty.docx");
    fs::write(&path, docx_bytes_with_text("")).unwrap();

    let reader = DocxReader::new();
    assert!(reader.read(&path).unwrap().is_none());
  }
}
