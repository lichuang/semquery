use thiserror::Error;

#[derive(Debug, Error)]
pub enum DocqError {
  #[error("parse error: {0}")]
  Parse(#[from] ParseError),

  #[error("store error: {0}")]
  Store(#[from] StoreError),

  #[error("embed error: {0}")]
  Embed(#[from] EmbedError),

  #[error("retrieve error: {0}")]
  Retrieve(#[from] RetrieveError),

  #[error("synth error: {0}")]
  Synth(#[from] SynthError),

  #[error("llm error: {0}")]
  Llm(#[from] LlmError),

  #[error("model error: {0}")]
  Model(#[from] ModelError),
}

pub type Result<T> = std::result::Result<T, DocqError>;

#[derive(Debug, Error)]
pub enum ParseError {
  #[error("read {path}: {source}")]
  Io { path: String, source: std::io::Error },

  #[error("extract {path}: {message}")]
  ExtractFailed { path: String, message: String },

  #[error("open docx {path}: {message}")]
  ZipFailed { path: String, message: String },

  #[error("zip entry `{entry}` missing in {path}: {message}")]
  ZipEntryMissing {
    path: String,
    entry: String,
    message: String,
  },

  #[error("parse document.xml: {message}")]
  XmlParseFailed { message: String },
}

#[derive(Debug, Error)]
pub enum StoreError {
  #[error("sqlite: {0}")]
  Sqlite(String),

  #[error("mutex poisoned")]
  MutexPoisoned,

  #[error("vector dimension must be greater than 0")]
  InvalidDimension,

  #[error("vec_chunks dimension mismatch: existing table does not use {expected}")]
  SchemaMismatch { expected: String },

  #[error("transaction already committed")]
  TransactionAlreadyCommitted,

  #[error("{what} length mismatch: {a} vs {b}")]
  ArgumentMismatch { what: String, a: usize, b: usize },

  #[error("io: {0}")]
  Io(String),

  #[error("collection not found: {0}")]
  NotFound(String),

  #[error("invalid timestamp: {0}")]
  InvalidTimestamp(String),
}

#[derive(Debug, Error)]
pub enum EmbedError {
  #[error("empty embedding result")]
  EmptyResult,

  #[error("mutex poisoned")]
  MutexPoisoned,

  #[error("inference failed: {0}")]
  InferenceFailed(String),
}

#[derive(Debug, Error)]
pub enum RetrieveError {
  #[error("task join: {0}")]
  TaskJoin(String),

  #[error("mutex poisoned")]
  MutexPoisoned,

  #[error("rerank failed: {0}")]
  RerankFailed(String),
}

#[derive(Debug, Error)]
pub enum SynthError {
  #[error("{0}")]
  Other(String),
}

#[derive(Debug, Error)]
pub enum LlmError {
  #[error("init backend: {0}")]
  BackendInit(String),

  #[error("load model: {0}")]
  ModelLoad(String),

  #[error("inference: {0}")]
  InferenceFailed(String),

  #[error("LLM not loaded — use open_for_ask")]
  NotLoaded,

  #[error("invalid config: {0}")]
  InvalidConfig(String),

  #[error("load tokenizer: {0}")]
  TokenizerLoad(String),
}

#[derive(Debug, Error)]
pub enum ModelError {
  #[error("model init failed: {0}")]
  ModelInitFailed(String),

  #[error("model info failed: {0}")]
  ModelInfoFailed(String),

  #[error("unsupported model: {0}")]
  UnsupportedModel(String),

  #[error("download failed: {0}")]
  DownloadFailed(String),

  #[error("hub api: {0}")]
  HubApiFailed(String),

  #[error("task join: {0}")]
  TaskJoin(String),
}
