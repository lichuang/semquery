use std::path::Path;

use semquery_core::{FileReader, ParseError, Result};

pub struct PdfReader;

impl Default for PdfReader {
  fn default() -> Self {
    Self::new()
  }
}

impl PdfReader {
  pub fn new() -> Self {
    Self
  }
}

impl FileReader for PdfReader {
  fn extensions(&self) -> &[&str] {
    &["pdf"]
  }

  fn read(&self, path: &Path) -> Result<Option<semquery_core::DocumentSource>> {
    let bytes = std::fs::read(path).map_err(|e| ParseError::Io {
      path: path.display().to_string(),
      source: e,
    })?;
    let text = pdf_extract::extract_text_from_mem(&bytes).map_err(|e| ParseError::ExtractFailed {
      path: path.display().to_string(),
      message: e.to_string(),
    })?;

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

#[cfg(test)]
mod tests {
  use super::*;
  use std::fs;
  use tempfile::TempDir;

  fn pdf_bytes_with_text(text: &str) -> Vec<u8> {
    use lopdf::content::{Content, Operation};
    use lopdf::{Document, Object, Stream, dictionary};

    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(dictionary! {
      "Type" => "Font",
      "Subtype" => "Type1",
      "BaseFont" => "Courier",
    });
    let resources_id = doc.add_object(dictionary! {
      "Font" => dictionary! { "F1" => font_id },
    });
    let content = Content {
      operations: vec![
        Operation::new("BT", vec![]),
        Operation::new("Tf", vec!["F1".into(), 12.into()]),
        Operation::new("Td", vec![100.into(), 700.into()]),
        Operation::new("Tj", vec![Object::string_literal(text)]),
        Operation::new("ET", vec![]),
      ],
    };
    let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
    let page_id = doc.add_object(dictionary! {
      "Type" => "Page",
      "Parent" => pages_id,
      "Contents" => content_id,
    });
    let pages = dictionary! {
      "Type" => "Pages",
      "Kids" => vec![page_id.into()],
      "Count" => 1,
      "Resources" => resources_id,
      "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));
    let catalog_id = doc.add_object(dictionary! {
      "Type" => "Catalog",
      "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);
    doc.compress();

    let mut buf = Vec::new();
    doc.save_to(&mut buf).unwrap();
    buf
  }

  fn empty_pdf_bytes() -> Vec<u8> {
    use lopdf::{Document, Object, dictionary};

    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let pages = dictionary! {
      "Type" => "Pages",
      "Kids" => Vec::<Object>::new(),
      "Count" => 0,
      "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));
    let catalog_id = doc.add_object(dictionary! {
      "Type" => "Catalog",
      "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    let mut buf = Vec::new();
    doc.save_to(&mut buf).unwrap();
    buf
  }

  #[test]
  fn test_pdf_reader_extracts_text() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("sample.pdf");
    fs::write(&path, pdf_bytes_with_text("Hello PDF")).unwrap();

    let reader = PdfReader::new();
    let doc = reader.read(&path).unwrap().unwrap();
    assert!(doc.content.contains("Hello PDF"));
  }

  #[test]
  fn test_pdf_reader_skips_empty_text() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("empty.pdf");
    fs::write(&path, empty_pdf_bytes()).unwrap();

    let reader = PdfReader::new();
    assert!(reader.read(&path).unwrap().is_none());
  }
}
