use std::ops::Range;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
  /// Stable document identifier; currently derived from the file path so the
  /// same path always maps to the same `id` across reindexes.
  pub id: String,
  /// SHA-256 of file content; drives incremental reindex.
  pub content_hash: String,
  pub content_size: usize,
  pub indexed_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
  /// SHA-256 of `text`; enables content-addressed dedup.
  pub id: String,
  /// `Document.id` this chunk belongs to.
  pub doc_id: String,
  /// Full original text actually embedded.
  pub text: String,
  pub byte_range: Range<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkCandidate {
  pub text: String,
  pub byte_range: Range<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
  pub chunk: Chunk,
  /// File system path of the source document, resolved from `document_paths`.
  pub file_path: PathBuf,
  pub score: f32,
  pub explain: ScoreExplain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchStage {
  EmbedQuery,
  Bm25Recall,
  VectorRecall,
  Fusion,
  Rerank,
  ResolvePaths,
  Assemble,
}

impl SearchStage {
  pub fn as_str(self) -> &'static str {
    match self {
      Self::EmbedQuery => "embed_query",
      Self::Bm25Recall => "bm25_recall",
      Self::VectorRecall => "vector_recall",
      Self::Fusion => "fusion",
      Self::Rerank => "rerank",
      Self::ResolvePaths => "resolve_paths",
      Self::Assemble => "assemble",
    }
  }
}

impl std::fmt::Display for SearchStage {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(self.as_str())
  }
}

/// Aggregated timings of one hybrid-search run, in milliseconds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchStats {
  pub total_ms: u64,
  pub embed_ms: u64,
  pub bm25_ms: u64,
  pub vector_ms: u64,
  pub fusion_ms: u64,
  pub rerank_ms: u64,
}

/// Streaming events emitted by the hybrid search pipeline, in order.
/// The stream ends after `Completed` or when the underlying future returns Err.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SearchEvent {
  /// A pipeline stage started — for progress display.
  StageStarted { stage: SearchStage },
  /// A pipeline stage finished — carries the stage duration.
  StageFinished { stage: SearchStage, elapsed_ms: u64 },
  /// Retrieval finished: final ranked hits. Also the terminal event.
  Completed { hits: Vec<SearchHit>, stats: SearchStats },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreExplain {
  pub bm25_score: Option<f32>,
  pub vector_score: Option<f32>,
  pub rrf_score: Option<f32>,
  pub rerank_score: Option<f32>,
  pub final_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredChunk {
  pub chunk: Chunk,
  pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
  pub text: String,
  pub citations: Vec<Citation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
  /// Marker as printed in `text`, e.g. `[1]`.
  pub marker: String,
  /// Human-readable source, e.g. `docs/a.txt (bytes 120-512)`.
  pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelRole {
  Embedding,
  Reranker,
  Chat,
  Tokenizer,
}

impl ModelRole {
  pub fn as_str(self) -> &'static str {
    match self {
      Self::Embedding => "embedding",
      Self::Reranker => "reranker",
      Self::Chat => "chat",
      Self::Tokenizer => "tokenizer",
    }
  }
}

impl std::fmt::Display for ModelRole {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(self.as_str())
  }
}

impl std::str::FromStr for ModelRole {
  type Err = String;
  fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
    match s {
      "embedding" => Ok(Self::Embedding),
      "reranker" => Ok(Self::Reranker),
      "chat" => Ok(Self::Chat),
      "tokenizer" => Ok(Self::Tokenizer),
      other => Err(format!("unknown model role: {other}")),
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
  pub role: ModelRole,
  pub repo_id: String,
  pub filename: String,
  pub revision: String,
  pub checksum: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
  pub n_ctx: u32,
  pub temperature: f32,
  pub top_p: f32,
  pub max_tokens: usize,
  pub seed: u32,
  pub system_prompt: String,
}

impl Default for LlmConfig {
  fn default() -> Self {
    Self {
      n_ctx: 8192,
      temperature: 0.7,
      top_p: 0.9,
      max_tokens: 512,
      seed: 0,
      system_prompt: "You are a helpful assistant.".into(),
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
  pub name: String,
  pub path: std::path::PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSource {
  pub path: std::path::PathBuf,
  pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatus {
  pub documents: usize,
  pub chunks: usize,
  pub collections: Vec<Collection>,
}
