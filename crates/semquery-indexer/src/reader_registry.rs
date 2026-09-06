use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use glob::Pattern;
use semquery_core::{FileReader, Result};
use walkdir::WalkDir;

/// Registry of file readers, keyed by file extension.
/// Holds one `Arc<dyn FileReader>` per registered reader, and dispatches
/// `read(path)` calls to the reader whose `extensions()` covers the file.
#[derive(Clone)]
pub struct ReaderRegistry {
  readers: Vec<Arc<dyn FileReader>>,
  ext_map: HashMap<String, usize>,
  ignore_patterns: Vec<Pattern>,
}

impl Default for ReaderRegistry {
  fn default() -> Self {
    Self::new()
  }
}

impl ReaderRegistry {
  pub fn new() -> Self {
    let ignore_patterns = [".git", "target", "node_modules"].iter().filter_map(|p| Pattern::new(p).ok()).collect();
    Self {
      readers: Vec::new(),
      ext_map: HashMap::new(),
      ignore_patterns,
    }
  }

  pub fn register(&mut self, reader: Arc<dyn FileReader>) {
    let idx = self.readers.len();
    for ext in reader.extensions() {
      self.ext_map.insert(ext.to_string(), idx);
    }
    self.readers.push(reader);
  }

  fn find_reader(&self, path: &Path) -> Option<&Arc<dyn FileReader>> {
    let ext = path.extension()?.to_str()?;
    let idx = self.ext_map.get(ext)?;
    self.readers.get(*idx)
  }

  fn is_ignored(&self, path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    name.starts_with('.') || self.ignore_patterns.iter().any(|pat| pat.matches(name))
  }

  /// Read a single file using the registered reader for its extension.
  /// Returns `None` if the path is ignored, has no registered reader, or the
  /// reader decides to skip the file (e.g. empty content).
  pub fn read_file(&self, path: &Path) -> Result<Option<semquery_core::DocumentSource>> {
    if self.is_ignored(path) {
      return Ok(None);
    }
    let reader = match self.find_reader(path) {
      Some(r) => r,
      None => return Ok(None),
    };
    reader.read(path)
  }

  /// Walk a directory and yield file paths that have a registered reader.
  /// Does NOT read file contents — caller reads on demand.
  pub fn list_files(&self, path: &Path, recursive: bool) -> Result<Vec<std::path::PathBuf>> {
    let walker = if recursive {
      WalkDir::new(path)
    } else {
      WalkDir::new(path).max_depth(1)
    };

    let mut paths = Vec::new();
    for entry in walker.into_iter().filter_entry(|e| e.depth() == 0 || !self.is_ignored(e.path())) {
      let entry = match entry {
        Ok(e) => e,
        Err(_) => continue,
      };
      if !entry.file_type().is_file() {
        continue;
      }
      if self.find_reader(entry.path()).is_some() {
        paths.push(entry.path().to_path_buf());
      }
    }
    Ok(paths)
  }

  /// Walk a directory, dispatch each file to the registered reader that
  /// handles its extension, and collect all `DocumentSource`s.
  pub fn read_dir(&self, path: &Path, recursive: bool) -> Result<Vec<semquery_core::DocumentSource>> {
    let mut docs = Vec::new();
    for p in self.list_files(path, recursive)? {
      if let Some(doc) = self.read_file(&p)? {
        docs.push(doc);
      }
    }
    Ok(docs)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::reader::TextFileReader;
  use std::fs;
  use tempfile::TempDir;

  fn default_registry() -> ReaderRegistry {
    let mut reg = ReaderRegistry::new();
    reg.register(Arc::new(TextFileReader::new()));
    reg
  }

  #[test]
  fn test_registry_txt_and_md() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("a.txt"), "hello").unwrap();
    fs::write(tmp.path().join("b.md"), "# world").unwrap();
    fs::write(tmp.path().join("c.bin"), "binary").unwrap();

    let reg = default_registry();
    let docs = reg.read_dir(tmp.path(), true).unwrap();
    assert_eq!(docs.len(), 2);
    let names: Vec<String> = docs.iter().map(|d| d.path.file_name().unwrap().to_str().unwrap().to_string()).collect();
    assert!(names.contains(&"a.txt".to_string()));
    assert!(names.contains(&"b.md".to_string()));
  }

  #[test]
  fn test_registry_ignores_hidden_and_dirs() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("visible.txt"), "ok").unwrap();
    fs::write(tmp.path().join(".hidden.txt"), "hidden").unwrap();

    let git_dir = tmp.path().join(".git");
    fs::create_dir_all(&git_dir).unwrap();
    fs::write(git_dir.join("config"), "git stuff").unwrap();

    let target_dir = tmp.path().join("target");
    fs::create_dir_all(&target_dir).unwrap();
    fs::write(target_dir.join("build.txt"), "build output").unwrap();

    let reg = default_registry();
    let docs = reg.read_dir(tmp.path(), true).unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].path.file_name().unwrap(), "visible.txt");
  }

  #[test]
  fn test_registry_recursive() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("top.txt"), "top").unwrap();
    let sub = tmp.path().join("sub");
    fs::create_dir_all(&sub).unwrap();
    fs::write(sub.join("deep.txt"), "deep").unwrap();

    let reg = default_registry();
    let docs_recursive = reg.read_dir(tmp.path(), true).unwrap();
    assert_eq!(docs_recursive.len(), 2);

    let docs_flat = reg.read_dir(tmp.path(), false).unwrap();
    assert_eq!(docs_flat.len(), 1);
  }

  #[test]
  fn test_registry_skips_non_utf8() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("bad.txt");
    let invalid_utf8: &[u8] = &[0xFF, 0xFE, 0xFD];
    fs::write(&path, invalid_utf8).unwrap();
    fs::write(tmp.path().join("good.txt"), "valid").unwrap();

    let reg = default_registry();
    let docs = reg.read_dir(tmp.path(), true).unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].path.file_name().unwrap(), "good.txt");
  }

  #[test]
  fn test_registry_skips_unregistered_extensions() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("a.txt"), "hello").unwrap();
    fs::write(tmp.path().join("b.pdf"), "fake pdf").unwrap();

    let reg = default_registry();
    let docs = reg.read_dir(tmp.path(), true).unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].path.file_name().unwrap(), "a.txt");
  }

  #[test]
  fn test_registry_multiple_readers() {
    struct PdfReader;
    impl FileReader for PdfReader {
      fn extensions(&self) -> &[&str] {
        &["pdf"]
      }
      fn read(&self, path: &Path) -> Result<Option<semquery_core::DocumentSource>> {
        Ok(Some(semquery_core::DocumentSource {
          path: path.to_path_buf(),
          content: "extracted pdf text".to_string(),
        }))
      }
    }

    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("a.txt"), "hello").unwrap();
    fs::write(tmp.path().join("b.pdf"), "%PDF-1.4 fake").unwrap();

    let mut reg = ReaderRegistry::new();
    reg.register(Arc::new(TextFileReader::new()));
    reg.register(Arc::new(PdfReader));

    let docs = reg.read_dir(tmp.path(), true).unwrap();
    assert_eq!(docs.len(), 2);
  }
}
