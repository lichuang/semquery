use std::collections::{HashMap, HashSet};
use std::ops::Add;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use rayon::prelude::*;
use semquery_core::{
  Chunk, Chunker, Document, EMBEDDING_BASELINE_KEY, Embedder, INDEXING_CONFIG_KEY, IndexEvent, ModelError, ModelRole,
  ModelSpec, Result, Storage, StoreError, Verbose, WordSegmenter,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::OnceCell;
use tokio::sync::mpsc::{self, Sender};
use tokio_stream::Stream;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::ReaderRegistry;

const EMBED_BATCH_SIZE: usize = 500;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IndexStats {
  pub files_indexed: usize,
  pub files_skipped: usize,
  pub files_removed: usize,
  pub chunks_indexed: usize,
}

impl Add for IndexStats {
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

type IndexEventItem = Result<IndexEvent>;
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
  /// Lazily-loaded embedder, shared with the engine. The engine fills the
  /// cell (eagerly today, on first index use after the lazy-loading
  /// refactor); the indexer only reads it.
  pub embedder: Arc<OnceCell<Arc<dyn Embedder>>>,
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
  embedder: Arc<OnceCell<Arc<dyn Embedder>>>,
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

  fn need_reindex(&self) -> Result<bool> {
    // Baseline is the spec recorded after the last *successful* index run
    // (`update_index_meta`), never the spec `ModelHub::ensure` writes on
    // model load — the latter is overwritten before any comparison happens,
    // which would turn the check into "new vs new".
    // `None` means no successful run was ever recorded: a fresh workspace
    // has nothing to re-embed, and pre-baseline databases (<= 0.4.1) whose
    // embeddings match live config must not be force-reindexed either.
    let model_changed = match self.storage.get_meta(EMBEDDING_BASELINE_KEY)? {
      None => false,
      Some(json) => serde_json::from_str::<ModelSpec>(&json)
        .map(|baseline| baseline != self.embedding_spec)
        .unwrap_or_else(|_| true),
    };
    if model_changed {
      return Ok(true);
    }
    let expected_chunk = format!("{}:{}", self.chunk_size, self.chunk_overlap);
    let chunk_changed = match self.storage.get_meta(INDEXING_CONFIG_KEY)? {
      Some(value) => value != expected_chunk,
      None => false,
    };
    Ok(chunk_changed)
  }

  fn update_index_meta(&self) -> Result<()> {
    self.storage.set_model_version_atomic(ModelRole::Embedding, &self.embedding_spec)?;
    let baseline = serde_json::to_string(&self.embedding_spec)
      .map_err(|e| StoreError::Io(format!("serialize embedding baseline: {e}")))?;
    self.storage.set_meta_atomic(EMBEDDING_BASELINE_KEY, &baseline)?;
    let indexing = format!("{}:{}", self.chunk_size, self.chunk_overlap);
    self.storage.set_meta_atomic(INDEXING_CONFIG_KEY, &indexing)?;
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
    let force_reindex = self.need_reindex()?;

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
        let stats = self.flush_batch(&mut batch, force_reindex, tx).await?;
        // Idempotent: also records meta on first-ever runs, so subsequent
        // runs can detect model / config changes.
        self.update_index_meta()?;
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

    let current_doc_ids: HashSet<String> = file_paths.iter().map(|p| sha256_hex(&p.to_string_lossy())).collect();

    let all_docs = self.storage.list_documents()?;
    let mut stats = IndexStats {
      files_removed: self.sweep_deleted(path, &current_doc_ids, &all_docs)?,
      ..Default::default()
    };

    let existing_map: Arc<HashMap<String, Document>> =
      Arc::new(all_docs.into_iter().map(|d| (d.id.clone(), d)).collect());

    let force_reindex = self.need_reindex()?;
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
        let s = self.flush_batch(&mut pending, force_reindex, tx).await?;
        stats.files_indexed += s.files_indexed;
        stats.chunks_indexed += s.chunks_indexed;
        pending_chunk_count = 0;
      }
    }

    let skipped = total - stats.files_indexed - pending.len();
    stats.files_skipped = skipped;

    if !pending.is_empty() {
      let s = self.flush_batch(&mut pending, force_reindex, tx).await?;
      stats.files_indexed += s.files_indexed;
      stats.chunks_indexed += s.chunks_indexed;
    }

    // Idempotent: also records meta on first-ever runs, so subsequent
    // runs can detect model / config changes.
    self.update_index_meta()?;

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

  fn sweep_deleted(&self, dir: &Path, current_doc_ids: &HashSet<String>, all_docs: &[Document]) -> Result<usize> {
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

  /// Embed all *new* chunks in `pending` in one batch, then write to storage
  /// in groups of `TX_BATCH_SIZE` files per transaction.
  ///
  /// Chunk ids are content hashes, so files sharing text (duplicated notes,
  /// boilerplate) collide with already-stored chunks. Those chunks are not
  /// re-embedded and not re-inserted — sqlite-vec `vec0` tables do not honor
  /// `INSERT OR REPLACE`, so a duplicate vector insert used to abort the
  /// whole index run. This document still links to the shared chunks via
  /// `chunk_documents`, keeping every file independently resolvable.
  ///
  /// When `reembed` is set (embedding model / indexing config changed), the
  /// filter is skipped and every chunk is re-embedded and overwritten.
  async fn flush_batch(
    &self,
    pending: &mut Vec<PendingFile>,
    reembed: bool,
    tx: &IndexEventSender,
  ) -> Result<IndexStats> {
    const TX_BATCH_SIZE: usize = 5;

    let mut existing: HashSet<String> = HashSet::new();
    if !reembed {
      let all_ids: Vec<String> = pending.iter().flat_map(|f| f.chunks.iter().map(|c| c.id.clone())).collect();
      existing = self.storage.get_existing_chunk_ids(&all_ids)?.into_iter().collect();
    }

    // Keep-set per file: chunks already in storage, or already claimed by an
    // earlier file in this same batch, are not embedded again.
    let mut claimed: HashSet<String> = HashSet::new();
    let plans: Vec<Vec<bool>> = pending
      .iter()
      .map(|pf| {
        pf.chunks
          .iter()
          .map(|c| reembed || (!existing.contains(&c.id) && claimed.insert(c.id.clone())))
          .collect()
      })
      .collect();

    let embed_texts: Vec<String> = pending
      .iter()
      .zip(&plans)
      .flat_map(|(pf, plan)| pf.chunk_texts.iter().zip(plan).filter(|(_, keep)| **keep).map(|(t, _)| t.clone()))
      .collect();
    if !send_event(
      tx,
      IndexEvent::EmbeddingBatch {
        count: embed_texts.len(),
      },
    )
    .await
    {
      return Ok(IndexStats::default());
    }
    let embedder = self.embedder.get().ok_or(ModelError::NotLoaded {
      component: "embedder",
      opener: "Engine::open",
    })?;
    let all_embeddings = embedder.embed(&embed_texts).await?;
    if !send_event(tx, IndexEvent::WritingStore).await {
      return Ok(IndexStats::default());
    }

    let mut stats = IndexStats::default();
    let mut offset = 0usize;
    let mut tx_count = 0usize;
    let mut tx = self.storage.begin_tx()?;

    for (pf, plan) in pending.drain(..).zip(plans) {
      let kept: Vec<usize> = plan.iter().enumerate().filter_map(|(i, keep)| keep.then_some(i)).collect();
      let new_chunks: Vec<Chunk> = kept.iter().map(|&i| pf.chunks[i].clone()).collect();
      let embeddings: Vec<Vec<f32>> = all_embeddings[offset..offset + kept.len()].to_vec();
      let new_ids: Vec<String> = new_chunks.iter().map(|c| c.id.clone()).collect();
      let new_tokenized: Vec<String> = kept.iter().map(|&i| pf.tokenized_texts[i].clone()).collect();
      let link_ids: Vec<String> = pf.chunks.iter().map(|c| c.id.clone()).collect();
      offset += kept.len();

      if pf.is_update {
        tx.unlink_chunks_by_doc(&pf.doc.id)?;
      }
      tx.add_document(&pf.doc)?;
      tx.set_document_path(&pf.doc.id, &pf.path.to_string_lossy())?;
      tx.add_chunks(&new_chunks)?;
      tx.add_chunk_documents(&link_ids, &pf.doc.id)?;
      tx.add_vectors(&new_ids, &embeddings)?;
      tx.add_fts_chunks(&new_ids, &new_tokenized)?;

      stats.files_indexed += 1;
      stats.chunks_indexed += kept.len();
      tx_count += 1;

      if tx_count >= TX_BATCH_SIZE {
        tx.commit()?;
        tx = self.storage.begin_tx()?;
        tx_count = 0;
      }
    }

    // Sweep only after every file in the batch has been linked: an update
    // unlinks its old chunks but a shared chunk may still be claimed by a
    // later file (or the updated file's own unchanged chunks).
    tx.sweep_orphan_chunks()?;
    tx.commit()?;

    Ok(stats)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{JiebaSegmenter, TextFileReader};
  use semquery_core::{ChunkCandidate, Embedder};
  use semquery_storage::SqliteStorage;
  use std::fs;
  use tempfile::TempDir;

  #[test]
  fn test_index_stats_and_event_json_roundtrip() {
    let stats = IndexStats {
      files_indexed: 3,
      files_skipped: 1,
      files_removed: 2,
      chunks_indexed: 42,
    };
    let json = serde_json::to_string(&stats).unwrap();
    assert_eq!(serde_json::from_str::<IndexStats>(&json).unwrap(), stats);

    let event = IndexEvent::Complete {
      files: 3,
      chunks: 42,
      files_skipped: 1,
      files_removed: 2,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert_eq!(serde_json::from_str::<IndexEvent>(&json).unwrap(), event);

    // Tagged-enum representation must also handle the unit variant.
    let event = IndexEvent::WritingStore;
    let json = serde_json::to_string(&event).unwrap();
    assert_eq!(serde_json::from_str::<IndexEvent>(&json).unwrap(), event);
  }

  fn stub_embedder_cell() -> Arc<OnceCell<Arc<dyn Embedder>>> {
    Arc::new(OnceCell::from(Arc::new(StubEmbedder { dim: 512 }) as Arc<dyn Embedder>))
  }

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
      embedder: stub_embedder_cell(),
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
    fs::write(&path, "hello world").unwrap();

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
    fs::write(&path, "hello world").unwrap();

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
    fs::write(&path, "hello world").unwrap();

    let storage = test_storage();
    let indexer = test_indexer(storage);

    let stats1 = indexer.index_file(&path).await.unwrap();
    assert_eq!(stats1.chunks_indexed, 1);

    fs::write(&path, "changed content").unwrap();
    let stats2 = indexer.index_file(&path).await.unwrap();
    assert_eq!(stats2.files_indexed, 1);
    assert_eq!(stats2.chunks_indexed, 1);
    assert_eq!(stats2.files_skipped, 0);
  }

  #[tokio::test]
  async fn test_index_directory() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("a.txt"), "first file").unwrap();
    fs::write(tmp.path().join("b.md"), "second file").unwrap();
    fs::write(tmp.path().join("c.bin"), "binary").unwrap();

    let storage = test_storage();
    let indexer = test_indexer(storage);

    let stats = indexer.index_directory(tmp.path()).await.unwrap();
    assert_eq!(stats.files_indexed, 2);
    assert_eq!(stats.chunks_indexed, 2);
  }

  // Issue #7 repro: two files with identical content must index successfully.
  // Their chunks share one content hash, so the vector insert used to hit a
  // primary-key conflict on the sqlite-vec vec0 table and abort the run.
  #[tokio::test]
  async fn test_index_duplicate_files_shared_chunk() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("a.md"), "Same sentence in two files.").unwrap();
    fs::write(tmp.path().join("b.md"), "Same sentence in two files.").unwrap();

    let storage = Arc::new(test_storage());
    let indexer = Indexer::new(IndexerConfig {
      chunker: Arc::new(StubChunker),
      embedder: stub_embedder_cell(),
      segmenter: Arc::new(JiebaSegmenter),
      storage: storage.clone(),
      readers: test_readers(),
      verbose: Verbose(false),
      embedding_spec: test_embedding_spec(),
      chunk_size: 1024,
      chunk_overlap: 102,
    });

    let stats = indexer.index_directory(tmp.path()).await.unwrap();
    assert_eq!(stats.files_indexed, 2);
    assert_eq!(stats.chunks_indexed, 1, "shared chunk must be embedded only once");

    assert_eq!(storage.list_documents().unwrap().len(), 2);
    assert_eq!(storage.count_chunks().unwrap(), 1);
    let doc_ids: Vec<String> = storage.list_documents().unwrap().into_iter().map(|d| d.id).collect();
    let paths = storage.get_document_paths(&doc_ids).unwrap();
    assert_eq!(paths.len(), 2, "both files must be independently resolvable");
  }

  // Same dedup, but the duplicate is discovered in a later indexing run
  // (the chunk is already committed to storage, not just claimed in-batch).
  #[tokio::test]
  async fn test_index_duplicate_files_across_runs() {
    let tmp = TempDir::new().unwrap();
    let a = tmp.path().join("a.md");
    let b = tmp.path().join("b.md");
    fs::write(&a, "dup content").unwrap();
    fs::write(&b, "dup content").unwrap();

    let storage = Arc::new(test_storage());
    let indexer = Indexer::new(IndexerConfig {
      chunker: Arc::new(StubChunker),
      embedder: stub_embedder_cell(),
      segmenter: Arc::new(JiebaSegmenter),
      storage: storage.clone(),
      readers: test_readers(),
      verbose: Verbose(false),
      embedding_spec: test_embedding_spec(),
      chunk_size: 1024,
      chunk_overlap: 102,
    });

    let stats1 = indexer.index_file(&a).await.unwrap();
    assert_eq!(stats1.chunks_indexed, 1);

    let stats2 = indexer.index_file(&b).await.unwrap();
    assert_eq!(stats2.files_indexed, 1);
    assert_eq!(stats2.chunks_indexed, 0, "already-stored chunk must not be re-embedded");

    assert_eq!(storage.count_chunks().unwrap(), 1);
    assert_eq!(storage.list_documents().unwrap().len(), 2);
  }

  #[tokio::test]
  async fn test_index_directory_removes_deleted_files() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("a.txt"), "first file").unwrap();
    fs::write(tmp.path().join("b.txt"), "second file").unwrap();

    let storage: Arc<dyn Storage> = Arc::new(test_storage());
    let indexer = Indexer::new(IndexerConfig {
      chunker: Arc::new(StubChunker),
      embedder: stub_embedder_cell(),
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

    fs::remove_file(tmp.path().join("a.txt")).unwrap();

    let indexer2 = Indexer::new(IndexerConfig {
      chunker: Arc::new(StubChunker),
      embedder: stub_embedder_cell(),
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
    fs::write(&path, "分布式共识算法").unwrap();

    let storage = Arc::new(test_storage());
    let indexer = Indexer::new(IndexerConfig {
      chunker: Arc::new(StubChunker),
      embedder: stub_embedder_cell(),
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
    fs::write(&path, "hello world").unwrap();

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
      embedder: stub_embedder_cell(),
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
      embedder: stub_embedder_cell(),
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

  struct LineChunker;

  impl Chunker for LineChunker {
    fn chunk(&self, text: &str) -> Vec<ChunkCandidate> {
      let mut chunks = Vec::new();
      let mut offset = 0usize;
      for line in text.split('\n') {
        if !line.trim().is_empty() {
          chunks.push(ChunkCandidate {
            text: line.to_string(),
            byte_range: offset..offset + line.len(),
          });
        }
        offset += line.len() + 1;
      }
      chunks
    }
  }

  fn issue8_indexer(storage: &Arc<SqliteStorage>) -> Indexer {
    Indexer::new(IndexerConfig {
      chunker: Arc::new(LineChunker),
      embedder: stub_embedder_cell(),
      segmenter: Arc::new(JiebaSegmenter),
      storage: storage.clone(),
      readers: test_readers(),
      verbose: Verbose(false),
      embedding_spec: test_embedding_spec(),
      chunk_size: 1024,
      chunk_overlap: 102,
    })
  }

  /// Regression test for issue #8 (secondary finding "model change still not
  /// detected"): `ModelHub::ensure` writes the freshly-loaded spec into
  /// `model_versions` BEFORE `need_reindex` compares against it, turning the
  /// check into "new vs new". `need_reindex` must therefore read the
  /// baseline recorded by `update_index_meta` after the last successful run,
  /// not `model_versions`.
  ///
  /// The overwrite done here via `set_model_version_atomic` mirrors exactly
  /// what the lazy-loading path does on first use.
  #[tokio::test]
  async fn test_need_reindex_ignores_model_versions_overwrite() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("note.txt");
    fs::write(&path, "hello world").unwrap();

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
      embedder: stub_embedder_cell(),
      segmenter: Arc::new(JiebaSegmenter),
      storage: storage.clone(),
      readers: test_readers(),
      verbose: Verbose(false),
      embedding_spec: spec_v1,
      chunk_size: 1024,
      chunk_overlap: 102,
    });
    assert!(!indexer_v1.need_reindex().unwrap(), "fresh workspace: no baseline yet");
    indexer_v1.index_file(&path).await.unwrap();
    assert!(!indexer_v1.need_reindex().unwrap(), "baseline now matches v1");

    // The lazy-loading path overwrites `model_versions` with the NEW spec
    // before any comparison can observe the old value.
    let spec_v2 = ModelSpec {
      role: ModelRole::Embedding,
      repo_id: "stub/embedding-v2".into(),
      filename: "model.onnx".into(),
      revision: "main".into(),
      checksum: None,
    };
    storage.set_model_version_atomic(ModelRole::Embedding, &spec_v2).unwrap();

    let indexer_v2 = Indexer::new(IndexerConfig {
      chunker: Arc::new(StubChunker),
      embedder: stub_embedder_cell(),
      segmenter: Arc::new(JiebaSegmenter),
      storage: storage.clone(),
      readers: test_readers(),
      verbose: Verbose(false),
      embedding_spec: spec_v2.clone(),
      chunk_size: 1024,
      chunk_overlap: 102,
    });
    assert!(
      indexer_v2.need_reindex().unwrap(),
      "model change must be detected even after model_versions was overwritten by ensure"
    );

    // After a successful v2 run the baseline catches up and the check goes quiet.
    let stats = indexer_v2.index_file(&path).await.unwrap();
    assert_eq!(stats.files_indexed, 1, "changed model forces re-embedding");
    assert!(!indexer_v2.need_reindex().unwrap(), "baseline now matches v2");
  }

  async fn drain_index(
    mut stream: impl Stream<Item = Result<IndexEvent>> + Unpin,
  ) -> (Option<IndexStats>, Option<String>) {
    let mut stats = None;
    let mut error = None;
    while let Some(event) = stream.next().await {
      match event {
        Ok(IndexEvent::Complete {
          files,
          chunks,
          files_skipped,
          files_removed,
        }) => {
          stats = Some(IndexStats {
            files_indexed: files,
            chunks_indexed: chunks,
            files_skipped,
            files_removed,
          });
        }
        Ok(IndexEvent::Error { message }) => error = Some(message),
        Ok(_) => {}
        Err(e) => error = Some(e.to_string()),
      }
    }
    (stats, error)
  }

  /// Regression test for https://github.com/lichuang/semquery/issues/8
  /// ("routine edits make `add` fail with `FOREIGN KEY constraint failed`").
  ///
  /// Repro: index a 3-line file (3 chunks), then append a 4th line. The first
  /// three chunks are unchanged, so `flush_batch` puts them in `existing` and
  /// only embeds the new one. The old code then called `delete_chunks_by_doc`,
  /// which both unlinked the document AND swept the chunks left unreferenced —
  /// deleting those three unchanged chunks before `add_chunk_documents` tried
  /// to re-link them. The re-link hit a dangling foreign key and the whole
  /// `add` run aborted. The fix splits unlink from sweep and defers the sweep
  /// until every file in the batch has been linked.
  #[tokio::test]
  async fn test_repro_issue_8_append_multi_chunk_file() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("note.md");
    fs::write(&path, "alpha line\nbeta line\ngamma line\n").unwrap();

    let storage = Arc::new(test_storage());
    let indexer = issue8_indexer(&storage);

    let (stats1, err1) =
      drain_index(indexer.index_sources_stream(vec![("docs".into(), tmp.path().to_path_buf())])).await;
    assert!(err1.is_none(), "first index must succeed, got: {err1:?}");
    assert_eq!(stats1.unwrap().chunks_indexed, 3, "3 lines -> 3 chunks");
    assert_eq!(storage.count_chunks().unwrap(), 3);

    fs::write(&path, "alpha line\nbeta line\ngamma line\ndelta line\n").unwrap();

    let (stats2, err2) =
      drain_index(indexer.index_sources_stream(vec![("docs".into(), tmp.path().to_path_buf())])).await;

    assert!(err2.is_none(), "append edit must not fail, got: {err2:?}");
    let stats2 = stats2.expect("append edit must emit Complete");
    assert_eq!(stats2.files_indexed, 1);
    assert_eq!(stats2.chunks_indexed, 1, "only the new line is embedded");
    assert_eq!(storage.count_chunks().unwrap(), 4, "3 unchanged + 1 new chunk");
    assert_eq!(storage.list_documents().unwrap().len(), 1);
  }

  #[tokio::test]
  async fn test_repro_issue_8_copy_and_rewrite_original() {
    let tmp = TempDir::new().unwrap();
    let report = tmp.path().join("report.md");
    fs::write(&report, "alpha line\nbeta line\n").unwrap();

    let storage = Arc::new(test_storage());
    let indexer = issue8_indexer(&storage);

    let (_stats, err) =
      drain_index(indexer.index_sources_stream(vec![("docs".into(), tmp.path().to_path_buf())])).await;
    assert!(err.is_none(), "first index must succeed, got: {err:?}");
    assert_eq!(storage.count_chunks().unwrap(), 2);

    // Keep the old version under a new name, then rewrite the original: the
    // copy still needs the old chunks while the original drops them.
    fs::copy(&report, tmp.path().join("old-report.md")).unwrap();
    fs::write(&report, "brand new wording\n").unwrap();

    let (stats, err) = drain_index(indexer.index_sources_stream(vec![("docs".into(), tmp.path().to_path_buf())])).await;

    assert!(err.is_none(), "copy + rewrite must not fail, got: {err:?}");
    assert_eq!(stats.expect("must emit Complete").files_indexed, 2);
    assert_eq!(storage.list_documents().unwrap().len(), 2);
    assert_eq!(
      storage.count_chunks().unwrap(),
      3,
      "2 old chunks kept by the copy + 1 new chunk for the rewritten original"
    );
  }
}
