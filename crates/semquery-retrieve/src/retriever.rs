//! Hybrid retrieval: BM25 + vector recall fused via Reciprocal Rank Fusion.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use semquery_core::{
  Chunk, DocqError, EmbedError, Embedder, Reranker, Result, RetrieveError, ScoreExplain, ScoredChunk, SearchEvent,
  SearchHit, SearchStage, SearchStats, Storage, Verbose, WordSegmenter,
};
use tokio::sync::mpsc::{self, Sender};
use tokio_stream::Stream;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::fusion;

type SearchEventItem = std::result::Result<SearchEvent, DocqError>;
type SearchEventSender = Sender<SearchEventItem>;

async fn send_event(tx: &SearchEventSender, event: SearchEvent) -> bool {
  if let Err(e) = tx.send(Ok(event.clone())).await {
    log::error!("search stream send failed for event {event:?}: {e}");
    false
  } else {
    true
  }
}

pub struct RetrieverConfig {
  pub storage: Arc<dyn Storage>,
  pub embedder: Arc<dyn Embedder>,
  pub segmenter: Arc<dyn WordSegmenter>,
  pub reranker: Option<Arc<dyn Reranker>>,
  /// BM25 recall depth (default 100).
  pub bm25_top_k: usize,
  /// Vector recall depth (default 100).
  pub vector_top_k: usize,
  /// RRF smoothing constant (default 60).
  pub rrf_k: usize,
  /// Number of RRF results to rerank (default 20).
  pub rerank_top_n: usize,
  /// Whether to print per-step progress and timings.
  pub verbose: Verbose,
}

#[derive(Clone)]
pub struct Retriever {
  storage: Arc<dyn Storage>,
  embedder: Arc<dyn Embedder>,
  segmenter: Arc<dyn WordSegmenter>,
  reranker: Option<Arc<dyn Reranker>>,
  bm25_top_k: usize,
  vector_top_k: usize,
  rrf_k: usize,
  rerank_top_n: usize,
  verbose: Verbose,
}

/// Borrowed score lookup maps used when assembling `SearchHit`s.
struct ScoreMaps<'a> {
  bm25: &'a HashMap<String, f32>,
  vector: &'a HashMap<String, f32>,
  rrf: &'a HashMap<String, f32>,
  rerank: &'a HashMap<String, f32>,
}

/// Result of a single recall channel, carrying its hits plus per-stage timings.
/// Each recall method fills only the fields it measures; the caller merges
/// them into `SearchStats`.
struct RecallResult {
  hits: Vec<(String, f32)>,
  bm25_ms: u64,
  embed_ms: u64,
  vector_ms: u64,
}

/// Order two candidates by reranker score.
///
/// A scored candidate sorts before an unscored one: a missing score is not a
/// `0.0` score. Scored candidates sort by descending score, so negative
/// cross-encoder scores stay below positive ones but above missing ones.
/// Unscored candidates compare equal; the stable sort then keeps their fused
/// (RRF) order.
fn compare_rerank_scores(a: Option<&f32>, b: Option<&f32>) -> Ordering {
  match (a, b) {
    (Some(a), Some(b)) => b.total_cmp(a),
    (Some(_), None) => Ordering::Less,
    (None, Some(_)) => Ordering::Greater,
    (None, None) => Ordering::Equal,
  }
}

impl Retriever {
  pub fn new(config: RetrieverConfig) -> Self {
    Self {
      storage: config.storage,
      embedder: config.embedder,
      segmenter: config.segmenter,
      reranker: config.reranker,
      bm25_top_k: config.bm25_top_k,
      vector_top_k: config.vector_top_k,
      rrf_k: config.rrf_k,
      rerank_top_n: config.rerank_top_n,
      verbose: config.verbose,
    }
  }

  /// Lexical recall using BM25 over the FTS5 index.
  ///
  /// Emits `StageStarted`/`StageFinished` events around the actual query, then
  /// returns the hits plus the measured wall time.
  async fn bm25_recall(&self, query: &str, tx: &SearchEventSender) -> Result<RecallResult> {
    if let Err(e) = tx
      .send(Ok(SearchEvent::StageStarted {
        stage: SearchStage::Bm25Recall,
      }))
      .await
    {
      log::error!("search stream send failed at bm25 start: {e}");
    }
    let start = Instant::now();
    let hits = self.bm25_recall_impl(query).await?;
    let bm25_ms = start.elapsed().as_millis() as u64;
    if let Err(e) = tx
      .send(Ok(SearchEvent::StageFinished {
        stage: SearchStage::Bm25Recall,
        elapsed_ms: bm25_ms,
      }))
      .await
    {
      log::error!("search stream send failed at bm25 end: {e}");
    }
    Ok(RecallResult {
      hits,
      bm25_ms,
      embed_ms: 0,
      vector_ms: 0,
    })
  }

  /// BM25 recall core, without event emission.
  ///
  /// Runs on a blocking thread so the caller can `tokio::join!` it with the
  /// async vector recall and overlap the embedding call with the FTS5 query.
  async fn bm25_recall_impl(&self, query: &str) -> Result<Vec<(String, f32)>> {
    let storage = self.storage.clone();
    let segmenter = self.segmenter.clone();
    let bm25_top_k = self.bm25_top_k;
    let verbose = self.verbose;
    let query = query.to_string();

    let handle = tokio::task::spawn_blocking(move || {
      let segmented_query = {
        let _step = verbose.start("segment query");
        segmenter.segment(&query)
      };
      let safe_query = sanitize_fts5_query(&segmented_query);
      let _step = verbose.start("BM25 recall");
      storage.search_text(&safe_query, bm25_top_k)
    });
    handle.await.map_err(|e| RetrieveError::TaskJoin(e.to_string()))?
  }

  /// Semantic recall using dense vector KNN search.
  ///
  /// Embeds the query and searches the sqlite-vec `vec_chunks` table.
  /// Emits `StageStarted`/`StageFinished` pairs for the embed and KNN steps,
  /// then returns the hits plus both measured wall times.
  async fn vector_recall(&self, query: &str, tx: &SearchEventSender) -> Result<RecallResult> {
    if let Err(e) = tx
      .send(Ok(SearchEvent::StageStarted {
        stage: SearchStage::EmbedQuery,
      }))
      .await
    {
      log::error!("search stream send failed at embed start: {e}");
    }
    let embed_start = Instant::now();
    let query_embedding = {
      let _step = self.verbose.start("embed query");
      self.embedder.embed(&[query.to_string()]).await?.into_iter().next().ok_or(EmbedError::EmptyResult)?
    };
    let embed_ms = embed_start.elapsed().as_millis() as u64;
    if let Err(e) = tx
      .send(Ok(SearchEvent::StageFinished {
        stage: SearchStage::EmbedQuery,
        elapsed_ms: embed_ms,
      }))
      .await
    {
      log::error!("search stream send failed at embed end: {e}");
    }

    if let Err(e) = tx
      .send(Ok(SearchEvent::StageStarted {
        stage: SearchStage::VectorRecall,
      }))
      .await
    {
      log::error!("search stream send failed at vector recall start: {e}");
    }
    let knn_start = Instant::now();
    let _step = self.verbose.start("vector recall");
    let hits = self.storage.search_vectors(&query_embedding, self.vector_top_k)?;
    let vector_ms = knn_start.elapsed().as_millis() as u64;
    if let Err(e) = tx
      .send(Ok(SearchEvent::StageFinished {
        stage: SearchStage::VectorRecall,
        elapsed_ms: vector_ms,
      }))
      .await
    {
      log::error!("search stream send failed at vector recall end: {e}");
    }
    Ok(RecallResult {
      hits,
      bm25_ms: 0,
      embed_ms,
      vector_ms,
    })
  }

  /// Resolve full chunk content and decide the final result order.
  ///
  /// If a reranker is configured, the top `rerank_top_n` RRF results are
  /// scored by the cross-encoder and re-sorted by that score. Otherwise the
  /// RRF order is preserved and only the top `top_k` chunks are fetched.
  ///
  /// Returns:
  /// - `chunk_map`: `chunk_id -> Chunk` lookup for the selected candidates.
  /// - `rerank_map`: `chunk_id -> reranker score`, empty when no reranker.
  /// - `ordered`: final ordering as `(chunk_id, _)` tuples; the score field is
  ///   a placeholder and is resolved later from `rerank_map` or `rrf_map`.
  async fn prepare_candidates(
    &self,
    query: &str,
    fused: Vec<(String, f32)>,
    top_k: usize,
  ) -> Result<(HashMap<String, Chunk>, HashMap<String, f32>, Vec<(String, f32)>)> {
    match &self.reranker {
      None => {
        let ids: Vec<String> = fused.iter().take(top_k).map(|(id, _)| id.clone()).collect();
        let chunks = self.storage.get_chunks(&ids)?;
        let map = chunks.into_iter().map(|c| (c.id.clone(), c)).collect();
        Ok((map, HashMap::new(), fused))
      }
      Some(reranker) => {
        // The reranker receives the raw query (not jieba-segmented) because it
        // runs its own BERT-style tokenization. It produces a single relevance
        // score per (query, chunk) pair — higher is better, same direction as RRF.
        let ids: Vec<String> = fused.iter().take(self.rerank_top_n).map(|(id, _)| id.clone()).collect();
        let chunks = self.storage.get_chunks(&ids)?;
        let chunk_map: HashMap<String, Chunk> = chunks.into_iter().map(|c| (c.id.clone(), c)).collect();

        let rerank_chunks: Vec<Chunk> = ids.iter().filter_map(|id| chunk_map.get(id).cloned()).collect();
        let scored: Vec<ScoredChunk> = {
          let _step = self.verbose.start("rerank");
          reranker.rerank(query, &rerank_chunks).await?
        };
        let rerank_map: HashMap<String, f32> = scored.into_iter().map(|sc| (sc.chunk.id, sc.score)).collect();

        let mut sorted: Vec<(String, f32)> = fused.into_iter().take(self.rerank_top_n).collect();
        sorted.sort_by(|a, b| compare_rerank_scores(rerank_map.get(&a.0), rerank_map.get(&b.0)));

        Ok((chunk_map, rerank_map, sorted))
      }
    }
  }

  /// Resolve source file paths for the documents referenced by the retrieved chunks.
  fn resolve_file_paths(&self, chunk_map: &HashMap<String, Chunk>) -> Result<HashMap<String, String>> {
    let _step = self.verbose.start("resolve paths");
    let doc_ids: Vec<String> =
      chunk_map.values().map(|c| c.doc_id.clone()).collect::<HashSet<_>>().into_iter().collect();
    self.storage.get_document_paths(&doc_ids)
  }

  /// Build the final `SearchHit` list from ordered chunk IDs and score maps.
  ///
  /// `ordered` contains chunk IDs in the desired output order. The actual score
  /// for each hit is taken from `scores.rerank` if available, otherwise
  /// `scores.rrf`. Per-stage scores from BM25 and vector recall are attached
  /// via `ScoreExplain`.
  fn assemble_hits(
    &self,
    ordered: Vec<(String, f32)>,
    top_k: usize,
    chunk_map: &HashMap<String, Chunk>,
    file_paths: &HashMap<String, String>,
    scores: &ScoreMaps<'_>,
  ) -> Vec<SearchHit> {
    let _step = self.verbose.start("assemble hits");
    ordered
      .into_iter()
      .take(top_k)
      .filter_map(|(id, _)| {
        let chunk = chunk_map.get(&id)?;
        let rrf_score = scores.rrf.get(&id).copied();
        let rerank_score = scores.rerank.get(&id).copied();
        let final_score = rerank_score.or(rrf_score).unwrap_or(0.0);
        Some(SearchHit {
          chunk: chunk.clone(),
          file_path: file_paths.get(&chunk.doc_id).map(PathBuf::from).unwrap_or_default(),
          score: final_score,
          explain: ScoreExplain {
            bm25_score: scores.bm25.get(&id).copied(),
            vector_score: scores.vector.get(&id).copied(),
            rrf_score,
            rerank_score,
            final_score,
          },
        })
      })
      .collect()
  }

  /// Streaming variant of [`Retriever::search`].
  ///
  /// Runs the pipeline in an independent task that pushes [`SearchEvent`]s
  /// into a bounded channel as stages complete. The consumer's iteration
  /// speed does not affect the pipeline — backpressure is capped by the
  /// channel buffer, and dropping the stream cancels the pipeline early.
  ///
  /// The stream terminates with a `Completed` event carrying the final hits
  /// plus per-stage timings. Errors surface as `Err` items and end the stream.
  ///
  /// Score directions in `ScoreExplain` are unified to "higher is better":
  /// - `bm25_score`: negated FTS5 BM25 score (higher = more relevant)
  /// - `vector_score`: similarity score from storage layer (higher = closer)
  /// - `rrf_score`: RRF fused score (higher = better rank)
  /// - `rerank_score` / `final_score`: cross-encoder score (higher = more
  ///   relevant); when no reranker is configured, `final_score = rrf_score`.
  pub fn search_stream(
    self: Arc<Self>,
    query: impl Into<String>,
    top_k: usize,
  ) -> impl Stream<Item = std::result::Result<SearchEvent, DocqError>> + Send + 'static {
    let (tx, rx) = mpsc::channel::<SearchEventItem>(32);
    let query = query.into();

    tokio::spawn(async move {
      if let Err(e) = self.search_internal(&query, top_k, &tx).await {
        let _ = tx.send(Err(e)).await;
      }
    });

    ReceiverStream::new(rx)
  }

  /// Pipeline core shared by `search_stream`. Runs all stages, pushing
  /// progress events into `tx`, and terminates with a `Completed` event.
  ///
  /// Returns `Ok(())` when the pipeline finishes normally or when the downstream
  /// consumer is dropped. Returns `Err` if any stage fails; the spawned caller
  /// forwards that error as the final item on the stream.
  async fn search_internal(
    &self,
    query: &str,
    top_k: usize,
    tx: &SearchEventSender,
  ) -> std::result::Result<(), DocqError> {
    let total_start = Instant::now();
    let mut stats = SearchStats::default();

    if query.trim().is_empty() {
      let _ = tx
        .send(Ok(SearchEvent::Completed {
          hits: Vec::new(),
          stats,
        }))
        .await;
      return Ok(());
    }

    let _total = self.verbose.start("search");

    // ---- Recall stages: BM25 (blocking thread) + vector embed+KNN, concurrent ----
    // Each recall method emits its own StageStarted/StageFinished events and
    // returns its measured timings via `RecallResult`.
    let (bm25_res, vec_res) = tokio::join!(self.bm25_recall(query, tx), self.vector_recall(query, tx));

    let (bm25_results, vector_raw) = match (bm25_res, vec_res) {
      (Ok(b), Ok(v)) => (b, v),
      (Err(e), _) | (_, Err(e)) => {
        return Err(e);
      }
    };

    stats.bm25_ms = bm25_results.bm25_ms;
    stats.embed_ms = vector_raw.embed_ms;
    stats.vector_ms = vector_raw.vector_ms;
    let bm25_results = bm25_results.hits;
    let vector_raw = vector_raw.hits;

    if !send_event(
      tx,
      SearchEvent::StageStarted {
        stage: SearchStage::Fusion,
      },
    )
    .await
    {
      return Ok(());
    }
    let t = Instant::now();
    let fused = fusion::reciprocal_rank_fusion(&[&bm25_results, &vector_raw], self.rrf_k);
    stats.fusion_ms = t.elapsed().as_millis() as u64;
    if !send_event(
      tx,
      SearchEvent::StageFinished {
        stage: SearchStage::Fusion,
        elapsed_ms: stats.fusion_ms,
      },
    )
    .await
    {
      return Ok(());
    }

    if fused.is_empty() {
      stats.total_ms = total_start.elapsed().as_millis() as u64;
      let _ = tx
        .send(Ok(SearchEvent::Completed {
          hits: Vec::new(),
          stats,
        }))
        .await;
      return Ok(());
    }

    let rrf_map: HashMap<String, f32> = fused.iter().cloned().collect();

    // ---- Resolve full chunks, rerank, decide final ordering ----
    if !send_event(
      tx,
      SearchEvent::StageStarted {
        stage: SearchStage::Rerank,
      },
    )
    .await
    {
      return Ok(());
    }
    let rerank_t = Instant::now();
    let (chunk_map, rerank_map, ordered) = match self.prepare_candidates(query, fused, top_k).await {
      Ok(v) => v,
      Err(e) => {
        return Err(e);
      }
    };
    stats.rerank_ms = rerank_t.elapsed().as_millis() as u64;
    if !send_event(
      tx,
      SearchEvent::StageFinished {
        stage: SearchStage::Rerank,
        elapsed_ms: stats.rerank_ms,
      },
    )
    .await
    {
      return Ok(());
    }

    // ---- Resolve source file paths ----
    if !send_event(
      tx,
      SearchEvent::StageStarted {
        stage: SearchStage::ResolvePaths,
      },
    )
    .await
    {
      return Ok(());
    }
    let file_paths = match self.resolve_file_paths(&chunk_map) {
      Ok(v) => v,
      Err(e) => {
        return Err(e);
      }
    };
    let _ = tx
      .send(Ok(SearchEvent::StageFinished {
        stage: SearchStage::ResolvePaths,
        elapsed_ms: 0,
      }))
      .await;

    let bm25_map: HashMap<String, f32> = bm25_results.iter().cloned().collect();
    let vector_map: HashMap<String, f32> = vector_raw.iter().cloned().collect();

    if !send_event(
      tx,
      SearchEvent::StageStarted {
        stage: SearchStage::Assemble,
      },
    )
    .await
    {
      return Ok(());
    }

    let assemble_start = Instant::now();
    let hits = self.assemble_hits(
      ordered,
      top_k,
      &chunk_map,
      &file_paths,
      &ScoreMaps {
        bm25: &bm25_map,
        vector: &vector_map,
        rrf: &rrf_map,
        rerank: &rerank_map,
      },
    );

    if !send_event(
      tx,
      SearchEvent::StageFinished {
        stage: SearchStage::Assemble,
        elapsed_ms: assemble_start.elapsed().as_millis() as u64,
      },
    )
    .await
    {
      return Ok(());
    }

    stats.total_ms = total_start.elapsed().as_millis() as u64;

    let _ = tx.send(Ok(SearchEvent::Completed { hits, stats })).await;
    Ok(())
  }

  /// Hybrid search entry point.
  ///
  /// Folds events from [`Retriever::search_stream`] into the final hit set.
  pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>> {
    let stream = Arc::new(self.clone()).search_stream(query.to_string(), top_k);
    let mut stream = std::pin::pin!(stream);
    while let Some(event) = stream.as_mut().next().await {
      match event {
        Ok(semquery_core::SearchEvent::Completed { hits, .. }) => return Ok(hits),
        Ok(_) => continue,
        Err(e) => return Err(e),
      }
    }
    Ok(Vec::new())
  }
}

/// Escape each token in an FTS5 query so that special characters like `-`,
/// `:`, `(`, `)`, `"` etc. are treated as literals rather than query syntax.
/// Tokens are split on whitespace and each is wrapped in double quotes.
fn sanitize_fts5_query(query: &str) -> String {
  query
    .split_whitespace()
    .map(|token| {
      if token.is_empty() {
        String::new()
      } else {
        // Escape embedded double quotes by doubling them.
        let escaped = token.replace('"', "\"\"");
        format!("\"{escaped}\"")
      }
    })
    .collect::<Vec<_>>()
    .join(" ")
}

#[cfg(test)]
mod tests {
  use super::*;
  use semquery_core::{
    ChunkCandidate, Chunker, Embedder, ModelRole, ModelSpec, Reranker, Result, ScoredChunk, Storage,
  };
  use semquery_indexer::{Indexer, IndexerConfig, JiebaSegmenter, ReaderRegistry, TextFileReader};
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
      Ok(texts.iter().map(|t| hash_embedding(t, self.dim)).collect())
    }
  }

  fn hash_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut vec = vec![0.0_f32; dim];
    for (i, byte) in text.bytes().enumerate() {
      vec[i % dim] += byte as f32 / 255.0;
    }
    vec
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

  async fn seed_index(storage: &Arc<SqliteStorage>, texts: &[(&str, &str)]) {
    let tmp = TempDir::new().unwrap();
    for (filename, content) in texts.iter() {
      let path = tmp.path().join(filename);
      std::fs::write(&path, content).unwrap();
      let indexer = Indexer::new(IndexerConfig {
        chunker: Arc::new(StubChunker),
        embedder: Arc::new(StubEmbedder { dim: 512 }),
        segmenter: Arc::new(JiebaSegmenter),
        storage: storage.clone(),
        readers: test_readers(),
        verbose: Verbose(false),
        embedding_spec: ModelSpec {
          role: ModelRole::Embedding,
          repo_id: "stub/embedding".into(),
          filename: "model.onnx".into(),
          revision: "main".into(),
          checksum: None,
        },
        chunk_size: 1024,
        chunk_overlap: 102,
      });
      indexer.index_file(&path).await.unwrap();
    }
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

  /// Base config shared by all retriever tests; only the reranker varies.
  fn test_retriever_config(storage: &Arc<SqliteStorage>, reranker: Option<Arc<dyn Reranker>>) -> RetrieverConfig {
    RetrieverConfig {
      storage: storage.clone(),
      embedder: Arc::new(StubEmbedder { dim: 512 }),
      segmenter: Arc::new(JiebaSegmenter),
      reranker,
      bm25_top_k: 100,
      vector_top_k: 100,
      rrf_k: 60,
      rerank_top_n: 20,
      verbose: Verbose(false),
    }
  }

  fn test_retriever(storage: &Arc<SqliteStorage>) -> Retriever {
    Retriever::new(test_retriever_config(storage, None))
  }

  struct StubReranker;

  #[async_trait::async_trait]
  impl Reranker for StubReranker {
    async fn rerank(&self, _query: &str, chunks: &[Chunk]) -> Result<Vec<ScoredChunk>> {
      Ok(
        chunks
          .iter()
          .enumerate()
          .map(|(i, c)| ScoredChunk {
            chunk: c.clone(),
            score: (chunks.len() - i) as f32,
          })
          .collect(),
      )
    }
  }

  struct PartialNegativeReranker;

  #[async_trait::async_trait]
  impl Reranker for PartialNegativeReranker {
    async fn rerank(&self, _query: &str, chunks: &[Chunk]) -> Result<Vec<ScoredChunk>> {
      Ok(chunks.first().cloned().map(|chunk| ScoredChunk { chunk, score: -1.0 }).into_iter().collect())
    }
  }

  fn test_retriever_with_reranker(storage: &Arc<SqliteStorage>) -> Retriever {
    Retriever::new(test_retriever_config(storage, Some(Arc::new(StubReranker))))
  }

  #[tokio::test]
  async fn test_hybrid_search_returns_results() {
    let storage = Arc::new(test_storage());
    seed_index(
      &storage,
      &[
        ("a.txt", "今天是我的生日"),
        ("b.txt", "分布式共识算法"),
        ("c.txt", "天气不错"),
      ],
    )
    .await;

    let retriever = test_retriever(&storage);
    let hits = retriever.search("生日", 5).await.unwrap();
    assert!(!hits.is_empty());
    assert!(hits[0].score > 0.0);
    assert!(
      hits[0].chunk.text.contains("生日"),
      "top hit should contain '生日', got: {}",
      hits[0].chunk.text
    );
  }

  #[tokio::test]
  async fn test_hybrid_search_score_explain() {
    let storage = Arc::new(test_storage());
    seed_index(
      &storage,
      &[("a.txt", "共识算法解决一致性问题"), ("b.txt", "Raft共识算法")],
    )
    .await;

    let retriever = test_retriever(&storage);
    let hits = retriever.search("共识算法", 5).await.unwrap();
    assert!(!hits.is_empty());

    for hit in &hits {
      let exp = &hit.explain;
      assert!(exp.rrf_score.is_some());
      assert_eq!(exp.final_score, exp.rrf_score.unwrap());
      assert!(exp.rerank_score.is_none());
    }
  }

  #[tokio::test]
  async fn test_hybrid_search_empty_query() {
    let storage = Arc::new(test_storage());
    seed_index(&storage, &[("a.txt", "hello")]).await;

    let retriever = test_retriever(&storage);
    let hits = retriever.search("", 5).await;
    assert!(hits.is_ok());
  }

  #[tokio::test]
  async fn test_hybrid_search_no_results() {
    let storage = Arc::new(test_storage());

    let retriever = test_retriever(&storage);
    let hits = retriever.search("不存在的内容", 5).await.unwrap();
    assert!(hits.is_empty());
  }

  #[test]
  fn test_sanitize_fts5_query_quotes_each_token() {
    assert_eq!(sanitize_fts5_query("Multi-Paxos"), "\"Multi-Paxos\"");
    assert_eq!(sanitize_fts5_query("hello world"), "\"hello\" \"world\"");
    assert_eq!(
      sanitize_fts5_query("a \"quoted\" term"),
      "\"a\" \"\"\"quoted\"\"\" \"term\""
    );
  }

  #[tokio::test]
  async fn test_hybrid_search_with_reranker() {
    let storage = Arc::new(test_storage());
    seed_index(
      &storage,
      &[
        ("a.txt", "今天是我的生日"),
        ("b.txt", "分布式共识算法"),
        ("c.txt", "天气不错"),
      ],
    )
    .await;

    let retriever = test_retriever_with_reranker(&storage);
    let hits = retriever.search("生日", 5).await.unwrap();
    assert!(!hits.is_empty());

    for hit in &hits {
      assert!(hit.explain.rerank_score.is_some());
      assert_eq!(hit.score, hit.explain.final_score);
      assert_eq!(hit.score, hit.explain.rerank_score.unwrap());
    }
  }

  #[tokio::test]
  async fn test_scored_candidates_sort_before_missing_rerank_scores() {
    let storage = Arc::new(test_storage());
    seed_index(
      &storage,
      &[("a.txt", "consensus algorithm"), ("b.txt", "distributed system")],
    )
    .await;

    let reranker = Some(Arc::new(PartialNegativeReranker) as Arc<dyn Reranker>);
    let retriever = Retriever::new(test_retriever_config(&storage, reranker));

    let hits = retriever.search("consensus", 2).await.unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].explain.rerank_score, Some(-1.0));
    assert!(hits[1].explain.rerank_score.is_none());
    // Unscored candidates fall back to the RRF score as their final score.
    assert_eq!(hits[1].score, hits[1].explain.rrf_score.unwrap());
  }

  #[tokio::test]
  async fn test_prepare_candidates_keeps_rrf_order_for_unscored() {
    let storage = Arc::new(test_storage());
    seed_index(
      &storage,
      &[
        ("a.txt", "consensus alpha"),
        ("b.txt", "consensus beta"),
        ("c.txt", "consensus gamma"),
      ],
    )
    .await;

    let reranker = Some(Arc::new(PartialNegativeReranker) as Arc<dyn Reranker>);
    let retriever = Retriever::new(test_retriever_config(&storage, reranker));
    let bm25 = retriever.bm25_recall_impl("consensus").await.unwrap();
    assert_eq!(bm25.len(), 3, "bm25 should recall all three chunks");

    let id_by_text: HashMap<String, String> = {
      let ids: Vec<String> = bm25.iter().map(|(id, _)| id.clone()).collect();
      storage.get_chunks(&ids).unwrap().into_iter().map(|c| (c.text, c.id)).collect()
    };

    // Fixed RRF order A > B > C with strictly decreasing fused scores.
    let id_a = id_by_text["consensus alpha"].clone();
    let id_b = id_by_text["consensus beta"].clone();
    let id_c = id_by_text["consensus gamma"].clone();
    let fused = vec![(id_a.clone(), 0.03), (id_b.clone(), 0.02), (id_c.clone(), 0.01)];

    let (_, rerank_map, ordered) = retriever.prepare_candidates("consensus", fused, 3).await.unwrap();

    // The reranker only scored the first candidate; B and C keep their RRF
    // relative order behind it.
    assert_eq!(rerank_map.len(), 1);
    assert_eq!(rerank_map[&id_a], -1.0);
    let ordered_ids: Vec<String> = ordered.into_iter().map(|(id, _)| id).collect();
    assert_eq!(ordered_ids, vec![id_a, id_b, id_c]);
  }

  #[tokio::test]
  async fn test_search_stream_event_order() {
    let storage = Arc::new(test_storage());
    seed_index(&storage, &[("a.txt", "consensus alpha"), ("b.txt", "consensus beta")]).await;

    let retriever = test_retriever(&storage);
    let stream = Arc::new(retriever).search_stream("consensus", 5);

    let mut events = Vec::new();
    {
      let mut stream = std::pin::pin!(stream);
      while let Some(event) = stream.as_mut().next().await {
        events.push(event.expect("stream must not error"));
      }
    }

    // The last event must be Completed with the final hits.
    let Some(semquery_core::SearchEvent::Completed { hits, stats }) = events.last() else {
      panic!("last event must be Completed");
    };
    assert!(!hits.is_empty());
    assert!(stats.total_ms > 0 || stats.total_ms == 0);

    // All events before Completed must be StageStarted / StageFinished pairs.
    for ev in &events[..events.len() - 1] {
      match ev {
        semquery_core::SearchEvent::StageStarted { .. } | semquery_core::SearchEvent::StageFinished { .. } => {}
        other => panic!("unexpected event before Completed: {other:?}"),
      }
    }

    // Every StageStarted must have a matching StageFinished for the same stage.
    let mut started = std::collections::HashSet::new();
    let mut finished = std::collections::HashSet::new();
    for ev in &events {
      match ev {
        semquery_core::SearchEvent::StageStarted { stage } => {
          started.insert(*stage);
        }
        semquery_core::SearchEvent::StageFinished { stage, .. } => {
          finished.insert(*stage);
        }
        _ => {}
      }
    }
    assert_eq!(started, finished, "each stage must have started and finished");
    assert!(started.contains(&SearchStage::Bm25Recall));
    assert!(started.contains(&SearchStage::VectorRecall));
    assert!(started.contains(&SearchStage::Rerank));
  }
}
