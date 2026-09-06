use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use rayon::prelude::*;
use semquery_core::{
  Chunk, Chunker, DocqError, Document, Embedder, IndexEvent, ModelRole, ModelSpec, Result, Storage, Verbose,
  WordSegmenter,
};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc::{self, Sender};
use tokio_stream::Stream;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::ReaderRegistry;

const EMBED_BATCH_SIZE: usize = 500;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct IndexStats {
  pub files_indexed: usize,
  pub files_skipped: usize,
  pub files_removed: usize,
  pub chunks_indexed: usize,
}

impl std::ops::Add for IndexStats {
  type Output = IndexStats;

  fn add(self, other: IndexStats) -> IndexStats {
    IndexStats {
      files_indexed: self.files_indexed + other.files_indexed,
      files_skipped: self.files_skipped + other.files_skipped,
      files_removed: self.files_removed + other.files_removed,
      chunks_indexed: self.chunks_indexed + other.chunks_indexed,
    }
  }
}

type IndexEventItem = std::result::Result<IndexEvent, DocqError>;
pub type IndexEventSender = Sender<IndexEventItem>;

async fn send_event(tx: &IndexEventSender, event: IndexEvent) -> bool {
  if let Err(e) = tx.send(Ok(event.clone())).await {
    log::error!("index stream send failed for event {event:?}: {e}");
    false
  } else {
    true
  }
}

fn sha256_hex(s: &str) -> String {
  let mut hasher = Sha256::new();
  hasher.update(s.as_bytes());
  let hash = hasher.finalize();
  const HEX: &[u8; 16] = b"0123456789abcdef";
  let mut out = String::with_capacity(64);
  for b in hash {
    out.push(HEX[(b >> 4) as usize] as char);
    out.push(HEX[(b & 0xf) as usize] as char);
  }
  out
}

pub struct IndexerConfig {
  pub chunker: Arc<dyn Chunker>,
  pub embedder: Arc<dyn Embedder>,
  pub segmenter: Arc<dyn WordSegmenter>,
  pub storage: Arc<dyn Storage>,
  pub readers: ReaderRegistry,
  pub verbose: Verbose,
  pub embedding_spec: ModelSpec,
  pub chunk_size: usize,
  pub chunk_overlap: usize,
}

#[derive(Clone)]
pub struct Indexer {
  chunker: Arc<dyn Chunker>,
  embedder: Arc<dyn Embedder>,
  segmenter: Arc<dyn WordSegmenter>,
  storage: Arc<dyn Storage>,
  readers: ReaderRegistry,
  verbose: Verbose,
  embedding_spec: ModelSpec,
  chunk_size: usize,
  chunk_overlap: usize,
}

struct PendingFile {
  path: PathBuf,
  doc: Document,
  chunks: Vec<Chunk>,
  chunk_texts: Vec<String>,
  tokenized_texts: Vec<String>,
  is_update: bool,
}

pub async fn collect_index_stats<S>(mut stream: S) -> Result<IndexStats>
where
  S: Stream<Item = Result<IndexEvent>> + Unpin,
{
  let mut stats = IndexStats::default();
  while let Some(event) = stream.next().await {
    if let IndexEvent::Complete {
      files,
      chunks,
      files_skipped,
      files_removed,
    } = event?
    {
      stats = IndexStats {
        files_indexed: files,
        chunks_indexed: chunks,
        files_skipped,
        files_removed,
      };
    }
  }
  Ok(stats)
}

impl Indexer {
  pub fn new(config: IndexerConfig) -> Self {
    Self {
      chunker: config.chunker,
      embedder: config.embedder,
      segmenter: config.segmenter,
      storage: config.storage,
      readers: config.readers,
      verbose: config.verbose,
      embedding_spec: config.embedding_spec,
      chunk_size: config.chunk_size,
      chunk_overlap: config.chunk_overlap,
    }
  }

  fn needs_reindex(&self) -> Result<bool> {
    let model_changed = match self.storage.get_model_version(ModelRole::Embedding)? {
      None => true,
      Some(spec) => {
        spec.repo_id != self.embedding_spec.repo_id
          || spec.filename != self.embedding_spec.filename
          || spec.revision != self.embedding_spec.revision
      }
    };
    if model_changed {
      return Ok(true);
    }
    let expected_chunk = format!("{}:{}", self.chunk_size, self.chunk_overlap);
    let chunk_changed = self.storage.get_meta("indexing")?.as_deref() != Some(expected_chunk.as_str());
    Ok(chunk_changed)
  }

  fn update_index_meta(&self) -> Result<()> {
    self.storage.set_model_version_atomic(ModelRole::Embedding, &self.embedding_spec)?;
    self.storage.set_meta_atomic("indexing", &format!("{}:{}", self.chunk_size, self.chunk_overlap))?;
    Ok(())
  }

  pub async fn index_file(&self, path: &Path) -> Result<IndexStats> {
    collect_index_stats(self.index_file_stream(path)).await
  }

  pub fn index_file_stream(&self, path: impl Into<PathBuf>) -> impl Stream<Item = Result<IndexEvent>> + Send + 'static {
    let (tx, rx) = mpsc::channel::<IndexEventItem>(32);
    let path = path.into();
    let this = <Self as Clone>::clone(self);
    tokio::spawn(async move {
      let _ = this.index_file_into_stream(&path, &tx).await;
    });
    ReceiverStream::new(rx)
  }

  pub async fn index_file_into_stream(&self, path: &Path, tx: &IndexEventSender) -> Result<IndexStats> {
    if !send_event(tx, IndexEvent::ScanStart { total_sources: 1 }).await {
      return Ok(IndexStats::default());
    }

    let doc_src = match self.readers.read_file(path)? {
      Some(doc) => doc,
      None => {
        return Ok(IndexStats {
          files_skipped: 1,
          ..Default::default()
        });
      }
    };

    let existing_docs = self.storage.list_documents()?;
    let existing_map: HashMap<String, Document> = existing_docs.into_iter().map(|d| (d.id.clone(), d)).collect();
    let force_reindex = self.needs_reindex()?;

    match self.prepare_file(&doc_src.path, &doc_src.content, &existing_map, force_reindex)? {
      Some(pending) => {
        let path = pending.path.to_string_lossy().to_string();
        if !send_event(
          tx,
          IndexEvent::FileParsed {
            source_id: path.clone(),
            path,
            chunks: pending.chunks.len(),
          },
        )
        .await
        {
          return Ok(IndexStats::default());
        }
        let mut batch = vec![pending];
        let stats = self.flush_batch(&mut batch, tx).await?;
        if force_reindex {
          self.update_index_meta()?;
        }
        let _ = send_event(
          tx,
          IndexEvent::Complete {
            files: stats.files_indexed,
            chunks: stats.chunks_indexed,
            files_skipped: 0,
            files_removed: 0,
          },
        )
        .await;
        Ok(stats)
      }
      None => {
        let _ = send_event(
          tx,
          IndexEvent::Complete {
            files: 0,
            chunks: 0,
            files_skipped: 1,
            files_removed: 0,
          },
        )
        .await;
        Ok(IndexStats {
          files_skipped: 1,
          ..Default::default()
        })
      }
    }
  }

  pub async fn index_directory(&self, path: &Path) -> Result<IndexStats> {
    collect_index_stats(self.index_directory_stream(path)).await
  }

  pub fn index_directory_stream(
    &self,
    path: impl Into<PathBuf>,
  ) -> impl Stream<Item = Result<IndexEvent>> + Send + 'static {
    let (tx, rx) = mpsc::channel::<IndexEventItem>(32);
    let path = path.into();
    let this = <Self as Clone>::clone(self);
    tokio::spawn(async move {
      let _ = this.index_directory_into_stream(&path, &tx).await;
    });
    ReceiverStream::new(rx)
  }

  pub fn index_sources_stream(
    &self,
    sources: Vec<(String, PathBuf)>,
  ) -> impl Stream<Item = Result<IndexEvent>> + Send + use<> + 'static {
    let (tx, rx) = mpsc::channel::<IndexEventItem>(32);
    let this = <Self as Clone>::clone(self);
    tokio::spawn(async move {
      let mut total_stats = IndexStats::default();
      for (source_id, path) in sources {
        match this.index_directory_into_stream(&path, &tx).await {
          Ok(stats) => total_stats = total_stats + stats,
          Err(e) => {
            let _ = send_event(&tx, IndexEvent::Error { message: e.to_string() }).await;
            return;
          }
        }
        if !send_event(
          &tx,
          IndexEvent::SourceComplete {
            source_id,
            chunks: total_stats.chunks_indexed,
          },
        )
        .await
        {
          return;
        }
      }
      let _ = send_event(
        &tx,
        IndexEvent::Complete {
          files: total_stats.files_indexed,
          chunks: total_stats.chunks_indexed,
          files_skipped: total_stats.files_skipped,
          files_removed: total_stats.files_removed,
        },
      )
      .await;
    });
    ReceiverStream::new(rx)
  }

  pub async fn index_directory_into_stream(&self, path: &Path, tx: &IndexEventSender) -> Result<IndexStats> {
    let file_paths = self.readers.list_files(path, true)?;
    let total = file_paths.len();
    if !send_event(tx, IndexEvent::ScanStart { total_sources: total }).await {
      return Ok(IndexStats::default());
    }

    let current_doc_ids: std::collections::HashSet<String> =
      file_paths.iter().map(|p| sha256_hex(&p.to_string_lossy())).collect();

    let all_docs = self.storage.list_documents()?;
    let mut stats = IndexStats {
      files_removed: self.sweep_deleted(path, &current_doc_ids, &all_docs)?,
      ..Default::default()
    };

    let existing_map: Arc<HashMap<String, Document>> =
      Arc::new(all_docs.into_iter().map(|d| (d.id.clone(), d)).collect());

    let force_reindex = self.needs_reindex()?;
    if force_reindex {
      self.verbose.log("indexing config or embedding model changed — forcing full re-index");
    }

    let existing_map = existing_map.clone();

    let prepared: Vec<Option<PendingFile>> = file_paths
      .par_iter()
      .enumerate()
      .map(|(i, file_path)| {
        if i % 10 == 0 {
          self.verbose.log(&format!(
            "chunking file {}/{} ({:.0}%): {}",
            i + 1,
            total,
            (i + 1) as f32 / total.max(1) as f32 * 100.0,
            file_path.display()
          ));
        }

        let doc_src = match self.readers.read_file(file_path) {
          Ok(Some(doc)) => doc,
          Ok(None) => return Ok(None),
          Err(e) => return Err(e),
        };

        self.prepare_file(&doc_src.path, &doc_src.content, &existing_map, force_reindex)
      })
      .collect::<Result<Vec<_>>>()?;

    let mut pending: Vec<PendingFile> = Vec::new();
    let mut pending_chunk_count = 0usize;

    for pf in prepared.into_iter().flatten() {
      let path = pf.path.to_string_lossy().to_string();
      if !send_event(
        tx,
        IndexEvent::FileParsed {
          source_id: path.clone(),
          path,
          chunks: pf.chunks.len(),
        },
      )
      .await
      {
        return Ok(stats);
      }
      pending_chunk_count += pf.chunks.len();
      pending.push(pf);
      if pending_chunk_count >= EMBED_BATCH_SIZE {
        let s = self.flush_batch(&mut pending, tx).await?;
        stats.files_indexed += s.files_indexed;
        stats.chunks_indexed += s.chunks_indexed;
        pending_chunk_count = 0;
      }
    }

    let skipped = total - stats.files_indexed - pending.len();
    stats.files_skipped = skipped;

    if !pending.is_empty() {
      let s = self.flush_batch(&mut pending, tx).await?;
      stats.files_indexed += s.files_indexed;
      stats.chunks_indexed += s.chunks_indexed;
    }

    if force_reindex {
      self.update_index_meta()?;
    }

    let _ = send_event(
      tx,
      IndexEvent::Complete {
        files: stats.files_indexed,
        chunks: stats.chunks_indexed,
        files_skipped: stats.files_skipped,
        files_removed: stats.files_removed,
      },
    )
    .await;
    Ok(stats)
  }

  fn sweep_deleted(
    &self,
    dir: &Path,
    current_doc_ids: &std::collections::HashSet<String>,
    all_docs: &[Document],
  ) -> Result<usize> {
    if all_docs.is_empty() {
      return Ok(0);
    }

    let all_doc_ids: Vec<String> = all_docs.iter().map(|d| d.id.clone()).collect();
    let paths = self.storage.get_document_paths(&all_doc_ids)?;

    let dir_prefix = dir.to_string_lossy().to_string();
    let mut removed = 0usize;

    for doc in all_docs {
      if current_doc_ids.contains(&doc.id) {
        continue;
      }
      let Some(file_path) = paths.get(&doc.id) else {
        continue;
      };
      if !file_path.starts_with(&dir_prefix) {
        continue;
      }
      let mut tx = self.storage.begin_tx()?;
      tx.delete_document(&doc.id)?;
      tx.delete_chunks_by_doc(&doc.id)?;
      tx.commit()?;
      removed += 1;
    }

    Ok(removed)
  }

  /// Read a single file's content, skip unchanged/empty files, and build a
  /// `PendingFile` ready for batched embedding and storage.
  fn prepare_file(
    &self,
    path: &Path,
    content: &str,
    existing_docs: &HashMap<String, Document>,
    force_reembed: bool,
  ) -> Result<Option<PendingFile>> {
    let content_hash = sha256_hex(content);
    let path_str = path.to_string_lossy().to_string();
    let doc_id = sha256_hex(&path_str);

    let is_update = if let Some(existing) = existing_docs.get(&doc_id) {
      if !force_reembed && existing.content_hash == content_hash {
        return Ok(None);
      }
      true
    } else {
      false
    };

    let candidates = self.chunker.chunk(content);
    if candidates.is_empty() {
      return Ok(None);
    }

    let chunk_texts: Vec<String> = candidates.iter().map(|c| c.text.clone()).collect();
    let tokenized_texts: Vec<String> = chunk_texts.iter().map(|t| self.segmenter.segment(t)).collect();
    let chunks: Vec<Chunk> = candidates
      .iter()
      .map(|c| Chunk {
        id: sha256_hex(&c.text),
        doc_id: doc_id.clone(),
        text: c.text.clone(),
        byte_range: c.byte_range.clone(),
      })
      .collect();

    let doc = Document {
      id: doc_id,
      content_hash,
      content_size: content.len(),
      indexed_at: Utc::now(),
    };

    Ok(Some(PendingFile {
      path: path.to_path_buf(),
      doc,
      chunks,
      chunk_texts,
      tokenized_texts,
      is_update,
    }))
  }

  /// Embed all chunks in `pending` in one batch, then write to storage
  /// in groups of `TX_BATCH_SIZE` files per transaction.
  async fn flush_batch(&self, pending: &mut Vec<PendingFile>, tx: &IndexEventSender) -> Result<IndexStats> {
    const TX_BATCH_SIZE: usize = 5;

    let all_texts: Vec<String> = pending.iter().flat_map(|f| f.chunk_texts.iter().cloned()).collect();
    if !send_event(tx, IndexEvent::EmbeddingBatch { count: all_texts.len() }).await {
      return Ok(IndexStats::default());
    }
    let all_embeddings = self.embedder.embed(&all_texts).await?;
    if !send_event(tx, IndexEvent::WritingStore).await {
      return Ok(IndexStats::default());
    }

    let mut stats = IndexStats::default();
    let mut offset = 0usize;
    let mut tx_count = 0usize;
    let mut tx = self.storage.begin_tx()?;

    for pf in pending.drain(..) {
      let n = pf.chunks.len();
      let embeddings: Vec<Vec<f32>> = all_embeddings[offset..offset + n].to_vec();
      let chunk_ids: Vec<String> = pf.chunks.iter().map(|c| c.id.clone()).collect();

      if pf.is_update {
        tx.delete_chunks_by_doc(&pf.doc.id)?;
      }
      tx.add_document(&pf.doc)?;
      tx.set_document_path(&pf.doc.id, &pf.path.to_string_lossy())?;
      tx.add_chunks(&pf.chunks)?;
      tx.add_chunk_documents(&chunk_ids, &pf.doc.id)?;
      tx.add_vectors(&chunk_ids, &embeddings)?;
      tx.add_fts_chunks(&chunk_ids, &pf.tokenized_texts)?;

      stats.files_indexed += 1;
      stats.chunks_indexed += n;
      offset += n;
      tx_count += 1;

      if tx_count >= TX_BATCH_SIZE {
        tx.commit()?;
        tx = self.storage.begin_tx()?;
        tx_count = 0;
      }
    }

    if tx_count > 0 {
      tx.commit()?;
    }

    Ok(stats)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{JiebaSegmenter, TextFileReader};
  use semquery_core::{ChunkCandidate, Embedder};
  use semquery_storage::SqliteStorage;
  use tempfile::TempDir;

  struct StubEmbedder {
    dim: usize,
  }

  #[async_trait::async_trait]
  impl Embedder for StubEmbedder {
    fn dimension(&self) -> usize {
      self.dim
    }

    fn model_name(&self) -> &str {
      "stub"
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
      Ok(texts.iter().map(|_| vec![0.1_f32; self.dim]).collect())
    }
  }

  struct StubChunker;

  impl Chunker for StubChunker {
    fn chunk(&self, text: &str) -> Vec<ChunkCandidate> {
      if text.trim().is_empty() {
        return Vec::new();
      }
      vec![ChunkCandidate {
        text: text.to_string(),
        byte_range: 0..text.len(),
      }]
    }
  }

  fn test_embedding_spec() -> ModelSpec {
    ModelSpec {
      role: ModelRole::Embedding,
      repo_id: "stub/embedding".into(),
      filename: "model.onnx".into(),
      revision: "main".into(),
      checksum: None,
    }
  }

  fn test_indexing_config() -> (usize, usize) {
    (1024, 102)
  }

  fn test_readers() -> ReaderRegistry {
    let mut reg = ReaderRegistry::new();
    reg.register(Arc::new(TextFileReader::new()));
    reg
  }

  fn test_storage() -> SqliteStorage {
    let s = SqliteStorage::open_in_memory().unwrap();
    s.init(512).unwrap();
    s
  }

  fn test_indexer(storage: SqliteStorage) -> Indexer {
    let (chunk_size, chunk_overlap) = test_indexing_config();
    Indexer::new(IndexerConfig {
      chunker: Arc::new(StubChunker),
      embedder: Arc::new(StubEmbedder { dim: 512 }),
      segmenter: Arc::new(JiebaSegmenter),
      storage: Arc::new(storage),
      readers: test_readers(),
      verbose: Verbose(false),
      embedding_spec: test_embedding_spec(),
      chunk_size,
      chunk_overlap,
    })
  }

  #[tokio::test]
  async fn test_index_file_basic() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("note.txt");
    std::fs::write(&path, "hello world").unwrap();

    let storage = test_storage();
    let indexer = test_indexer(storage);

    let stats = indexer.index_file(&path).await.unwrap();
    assert_eq!(stats.files_indexed, 1);
    assert_eq!(stats.chunks_indexed, 1);
    assert_eq!(stats.files_skipped, 0);
  }

  #[tokio::test]
  async fn test_index_file_skip_unchanged() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("note.txt");
    std::fs::write(&path, "hello world").unwrap();

    let storage = test_storage();
    let indexer = test_indexer(storage);

    let stats1 = indexer.index_file(&path).await.unwrap();
    assert_eq!(stats1.chunks_indexed, 1);

    let stats2 = indexer.index_file(&path).await.unwrap();
    assert_eq!(stats2.files_skipped, 1);
    assert_eq!(stats2.chunks_indexed, 0);
  }

  #[tokio::test]
  async fn test_index_file_reindex_on_change() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("note.txt");
    std::fs::write(&path, "hello world").unwrap();

    let storage = test_storage();
    let indexer = test_indexer(storage);

    let stats1 = indexer.index_file(&path).await.unwrap();
    assert_eq!(stats1.chunks_indexed, 1);

    std::fs::write(&path, "changed content").unwrap();
    let stats2 = indexer.index_file(&path).await.unwrap();
    assert_eq!(stats2.files_indexed, 1);
    assert_eq!(stats2.chunks_indexed, 1);
    assert_eq!(stats2.files_skipped, 0);
  }

  #[tokio::test]
  async fn test_index_directory() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.txt"), "first file").unwrap();
    std::fs::write(tmp.path().join("b.md"), "second file").unwrap();
    std::fs::write(tmp.path().join("c.bin"), "binary").unwrap();

    let storage = test_storage();
    let indexer = test_indexer(storage);

    let stats = indexer.index_directory(tmp.path()).await.unwrap();
    assert_eq!(stats.files_indexed, 2);
    assert_eq!(stats.chunks_indexed, 2);
  }

  #[tokio::test]
  async fn test_index_directory_removes_deleted_files() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.txt"), "first file").unwrap();
    std::fs::write(tmp.path().join("b.txt"), "second file").unwrap();

    let storage: Arc<dyn Storage> = Arc::new(test_storage());
    let indexer = Indexer::new(IndexerConfig {
      chunker: Arc::new(StubChunker),
      embedder: Arc::new(StubEmbedder { dim: 512 }),
      segmenter: Arc::new(JiebaSegmenter),
      storage: storage.clone(),
      readers: test_readers(),
      verbose: Verbose(false),
      embedding_spec: test_embedding_spec(),
      chunk_size: 1024,
      chunk_overlap: 102,
    });
    indexer.index_directory(tmp.path()).await.unwrap();

    assert_eq!(storage.list_documents().unwrap().len(), 2);

    std::fs::remove_file(tmp.path().join("a.txt")).unwrap();

    let indexer2 = Indexer::new(IndexerConfig {
      chunker: Arc::new(StubChunker),
      embedder: Arc::new(StubEmbedder { dim: 512 }),
      segmenter: Arc::new(JiebaSegmenter),
      storage: storage.clone(),
      readers: test_readers(),
      verbose: Verbose(false),
      embedding_spec: test_embedding_spec(),
      chunk_size: 1024,
      chunk_overlap: 102,
    });
    let stats = indexer2.index_directory(tmp.path()).await.unwrap();
    assert_eq!(stats.files_removed, 1);
    assert_eq!(stats.files_indexed, 0);

    let docs = storage.list_documents().unwrap();
    assert_eq!(docs.len(), 1);
  }

  #[tokio::test]
  async fn test_index_then_search() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("note.txt");
    std::fs::write(&path, "分布式共识算法").unwrap();

    let storage = Arc::new(test_storage());
    let indexer = Indexer::new(IndexerConfig {
      chunker: Arc::new(StubChunker),
      embedder: Arc::new(StubEmbedder { dim: 512 }),
      segmenter: Arc::new(JiebaSegmenter),
      storage: storage.clone(),
      readers: test_readers(),
      verbose: Verbose(false),
      embedding_spec: test_embedding_spec(),
      chunk_size: 1024,
      chunk_overlap: 102,
    });

    indexer.index_file(&path).await.unwrap();

    let hits = storage.search_text("共识 算法", 10).unwrap();
    assert_eq!(hits.len(), 1);
  }

  #[tokio::test]
  async fn test_model_upgrade_triggers_reembed() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("note.txt");
    std::fs::write(&path, "hello world").unwrap();

    let storage = Arc::new(test_storage());

    let spec_v1 = ModelSpec {
      role: ModelRole::Embedding,
      repo_id: "stub/embedding-v1".into(),
      filename: "model.onnx".into(),
      revision: "main".into(),
      checksum: None,
    };
    let indexer_v1 = Indexer::new(IndexerConfig {
      chunker: Arc::new(StubChunker),
      embedder: Arc::new(StubEmbedder { dim: 512 }),
      segmenter: Arc::new(JiebaSegmenter),
      storage: storage.clone(),
      readers: test_readers(),
      verbose: Verbose(false),
      embedding_spec: spec_v1,
      chunk_size: 1024,
      chunk_overlap: 102,
    });
    let stats1 = indexer_v1.index_file(&path).await.unwrap();
    assert_eq!(stats1.files_indexed, 1);

    let stats2 = indexer_v1.index_file(&path).await.unwrap();
    assert_eq!(stats2.files_skipped, 1, "same model — file should be skipped");

    let spec_v2 = ModelSpec {
      role: ModelRole::Embedding,
      repo_id: "stub/embedding-v2".into(),
      filename: "model.onnx".into(),
      revision: "main".into(),
      checksum: None,
    };
    let indexer_v2 = Indexer::new(IndexerConfig {
      chunker: Arc::new(StubChunker),
      embedder: Arc::new(StubEmbedder { dim: 512 }),
      segmenter: Arc::new(JiebaSegmenter),
      storage: storage.clone(),
      readers: test_readers(),
      verbose: Verbose(false),
      embedding_spec: spec_v2,
      chunk_size: 1024,
      chunk_overlap: 102,
    });
    let stats3 = indexer_v2.index_file(&path).await.unwrap();
    assert_eq!(stats3.files_indexed, 1, "model changed — file must be re-embedded");

    let recorded = storage.get_model_version(ModelRole::Embedding).unwrap().unwrap();
    assert_eq!(recorded.repo_id, "stub/embedding-v2");
  }
}
