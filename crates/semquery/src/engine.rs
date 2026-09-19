//! Engine facade — assembles all components into a single API.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use semquery_core::{
  Chunker, Collection, Embedder, EngineStatus, IndexEvent, Llm, LlmConfig, ModelRole, ModelSpec, Reranker, Result,
  SearchEvent, SearchHit, Storage, Verbose, WordSegmenter,
};
#[cfg(feature = "docx")]
use semquery_indexer::DocxReader;
#[cfg(feature = "pdf")]
use semquery_indexer::PdfReader;
use semquery_indexer::{
  IndexStats, Indexer, IndexerConfig, JiebaSegmenter, ReaderRegistry, SentenceSplitter, TextFileReader,
  collect_index_stats,
};
use semquery_model::{FastEmbedEmbedder, FastEmbedReranker, GgufLlm, ModelHub};
use semquery_retrieve::{Retriever, RetrieverConfig};
use tokio::sync::OnceCell;
use tokio::sync::mpsc;
use tokio_stream::Stream;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::config::{RetrievalConfig, SemqConfig};
use semquery_storage::SqliteStorage;
use semquery_synth::{Synthesizer, SynthesizerConfig};

pub struct EngineConfig {
  pub workspace_path: PathBuf,
  pub model_cache_dir: PathBuf,
  pub config: SemqConfig,
  pub verbose: Verbose,
}

/// Internal download-progress signal, mapped to the concrete event type
/// (`IndexEvent` / `SearchEvent` / `AskEvent`) by each streaming API.
enum DownloadPhase {
  Start,
  Complete { elapsed_ms: u64 },
}

/// Pre-built components for `Engine::new` (dependency injection).
/// Tests construct this with stub implementations; production code uses
/// [`Engine::open`] and lazy loading instead.
pub struct EngineComponents {
  pub storage: Arc<dyn Storage>,
  /// Model hub + config are kept so the engine can lazily load models on
  /// first use (see the lazy-loading refactor); `Engine::new` pre-fills the
  /// OnceCells below from these components.
  pub hub: ModelHub,
  pub config: SemqConfig,
  pub chunker: Arc<dyn Chunker>,
  pub embedder: Arc<dyn Embedder>,
  pub segmenter: Arc<dyn WordSegmenter>,
  pub reranker: Option<Arc<dyn Reranker>>,
  pub llm: Option<Arc<dyn Llm>>,
  pub readers: ReaderRegistry,
  pub retrieval: RetrievalConfig,
  pub verbose: Verbose,
  pub embedding_spec: ModelSpec,
  pub chunk_size: usize,
  pub chunk_overlap: usize,
}

pub struct Engine {
  storage: Arc<dyn Storage>,
  hub: ModelHub,
  config: SemqConfig,
  embedder: Arc<OnceCell<Arc<dyn Embedder>>>,
  reranker: Arc<OnceCell<Arc<dyn Reranker>>>,
  llm: Arc<OnceCell<Arc<dyn Llm>>>,
  /// Tokenizer-backed chunker, produced alongside the embedder (the chunker's
  /// tokenizer file lives in the embedding model repo).
  chunker: Arc<OnceCell<Arc<dyn Chunker>>>,
  indexer: Option<Arc<Indexer>>,
  retriever: Option<Arc<Retriever>>,
  synthesizer: Option<Synthesizer>,
  verbose: Verbose,
}

impl Engine {
  pub fn new(components: EngineComponents) -> Self {
    let EngineComponents {
      storage,
      hub,
      config,
      chunker,
      embedder,
      segmenter,
      reranker,
      llm,
      readers,
      retrieval,
      verbose,
      embedding_spec,
      chunk_size,
      chunk_overlap,
    } = components;

    // Wrap the injected models in shareable OnceCells: the engine keeps one
    // handle and each downstream component (indexer / retriever /
    // synthesizer) gets a clone, so later steps can drive loading from the
    // stream layer and have every component see the same instance.
    let embedder_cell = Arc::new(OnceCell::from(embedder));
    let reranker_cell = reranker.map(|r| Arc::new(OnceCell::from(r)));
    let llm_cell = llm.map(|l| Arc::new(OnceCell::from(l)));
    let chunker_cell = Arc::new(OnceCell::from(chunker.clone()));

    let indexer = Indexer::new(IndexerConfig {
      chunker,
      embedder: embedder_cell.clone(),
      segmenter: segmenter.clone(),
      storage: storage.clone(),
      readers,
      verbose,
      embedding_spec,
      chunk_size,
      chunk_overlap,
    });

    let retriever = Arc::new(Retriever::new(RetrieverConfig {
      storage: storage.clone(),
      embedder: embedder_cell.clone(),
      segmenter,
      reranker: reranker_cell.clone(),
      bm25_top_k: retrieval.bm25_top_k,
      vector_top_k: retrieval.vector_top_k,
      rrf_k: retrieval.rrf_k,
      rerank_top_n: retrieval.rerank_top_n,
      verbose,
    }));

    let synthesizer = llm_cell.clone().map(|cell| {
      Synthesizer::new(SynthesizerConfig {
        retriever: retriever.clone(),
        llm: cell,
        verbose,
      })
    });

    Self {
      storage,
      hub,
      config,
      embedder: embedder_cell,
      reranker: reranker_cell.unwrap_or_else(|| Arc::new(OnceCell::new())),
      llm: llm_cell.unwrap_or_else(|| Arc::new(OnceCell::new())),
      chunker: chunker_cell,
      indexer: Some(Arc::new(indexer)),
      retriever: Some(retriever),
      synthesizer,
      verbose,
    }
  }

  fn default_readers() -> ReaderRegistry {
    let mut reg = ReaderRegistry::new();
    reg.register(Arc::new(TextFileReader::new()));
    #[cfg(feature = "pdf")]
    reg.register(Arc::new(PdfReader::new()));
    #[cfg(feature = "docx")]
    reg.register(Arc::new(DocxReader::new()));
    reg
  }

  // ---- Shared helpers for model loading ----

  fn open_storage(workspace_path: &Path) -> Result<Arc<dyn Storage>> {
    std::fs::create_dir_all(workspace_path)
      .map_err(|e| semquery_core::StoreError::Io(format!("create workspace dir: {e}")))?;
    let storage: Arc<dyn Storage> = Arc::new(SqliteStorage::open_workspace(workspace_path)?);
    Ok(storage)
  }

  async fn build_chunker(
    hub: &ModelHub,
    emb_spec: &ModelSpec,
    tokenizer_filename: &str,
    indexing: &crate::config::IndexingConfig,
  ) -> Result<Arc<dyn Chunker>> {
    let tokenizer_spec = ModelSpec {
      role: ModelRole::Tokenizer,
      repo_id: emb_spec.repo_id.clone(),
      filename: tokenizer_filename.into(),
      revision: emb_spec.revision.clone(),
      checksum: None,
    };
    let path = hub.resolve(&tokenizer_spec).await?;
    let tokenizer =
      tokenizers::Tokenizer::from_file(&path).map_err(|e| semquery_core::LlmError::TokenizerLoad(e.to_string()))?;
    Ok(Arc::new(SentenceSplitter::new(
      tokenizer,
      indexing.chunk_size,
      indexing.chunk_overlap,
    )))
  }

  // ---- Open methods ----

  /// Open an engine without loading any models: no network access, no
  /// model-cache resolution. This is the canonical entry point —
  /// storage-level operations (`status`, `list_collections`,
  /// `add_collection`, `remove_collection`) work immediately, and
  /// model-backed operations (`index`, `search`, `ask`, plus their streaming
  /// variants) lazily load the models they need on first use, emitting
  /// `ModelDownloadStart` / `ModelDownloadComplete` events when a model file
  /// actually hits the network.
  pub fn open(config: EngineConfig) -> Result<Self> {
    let storage = Self::open_storage(&config.workspace_path)?;
    storage.init(0)?;
    Ok(Self {
      storage,
      hub: ModelHub::new(config.model_cache_dir.clone()),
      config: config.config.clone(),
      embedder: Arc::new(OnceCell::new()),
      reranker: Arc::new(OnceCell::new()),
      llm: Arc::new(OnceCell::new()),
      chunker: Arc::new(OnceCell::new()),
      indexer: None,
      retriever: None,
      synthesizer: None,
      verbose: config.verbose,
    })
  }

  pub fn add_collection(&self, name: &str, path: impl AsRef<Path>) -> Result<()> {
    let canonical = std::fs::canonicalize(path.as_ref())
      .map_err(|e| semquery_core::StoreError::Io(format!("canonicalize {}: {e}", path.as_ref().display())))?;
    let path_str = canonical.to_string_lossy().to_string();
    let mut tx = self.storage.begin_tx()?;
    tx.add_collection(name, &path_str)?;
    tx.commit()?;
    Ok(())
  }

  pub fn list_collections(&self) -> Result<Vec<Collection>> {
    self.storage.list_collections()
  }

  pub fn add_file(&self, name: &str, path: impl AsRef<Path>) -> Result<()> {
    let canonical = std::fs::canonicalize(path.as_ref())
      .map_err(|e| semquery_core::StoreError::Io(format!("canonicalize {}: {e}", path.as_ref().display())))?;
    if !canonical.is_file() {
      return Err(semquery_core::StoreError::Io(format!("{} is not a file", canonical.display())).into());
    }
    self.add_collection(name, &canonical)
  }

  pub fn remove_collection(&self, name: &str) -> Result<()> {
    let mut tx = self.storage.begin_tx()?;
    tx.delete_collection(name)?;
    tx.commit()?;
    Ok(())
  }

  pub fn clear_collections(&self) -> Result<()> {
    let mut tx = self.storage.begin_tx()?;
    tx.clear_collections()?;
    tx.commit()?;
    Ok(())
  }

  pub async fn index(&self) -> Result<IndexStats> {
    let _total = self.verbose.start("index");
    collect_index_stats(self.index_stream()?).await
  }

  pub fn index_stream(&self) -> Result<impl Stream<Item = Result<IndexEvent>> + Send + 'static> {
    let collections = self.storage.list_collections()?;
    let sources = collections.into_iter().map(|c| (c.name, c.path)).collect();
    Ok(self.spawn_index(sources))
  }

  pub async fn index_one(&self, name: &str) -> Result<IndexStats> {
    let _total = self.verbose.start("index one collection");
    collect_index_stats(self.index_one_stream(name)?).await
  }

  pub fn index_one_stream(
    &self,
    name: impl Into<String>,
  ) -> Result<impl Stream<Item = Result<IndexEvent>> + Send + 'static> {
    let name = name.into();
    let collections = self.storage.list_collections()?;
    let col = collections.into_iter().find(|c| c.name == name).ok_or(semquery_core::StoreError::NotFound(name))?;
    Ok(self.spawn_index(vec![(col.name, col.path)]))
  }

  /// Spawn an index pipeline. If the engine was built by `Engine::new` with a
  /// pre-built indexer, that indexer is used directly; otherwise the embedding
  /// model (and its tokenizer-backed chunker) is loaded lazily — emitting
  /// `ModelDownloadStart` / `ModelDownloadComplete` events when a model file
  /// actually hits the network — and an `Indexer` is constructed on first use.
  fn spawn_index(&self, sources: Vec<(String, PathBuf)>) -> impl Stream<Item = Result<IndexEvent>> + Send + 'static {
    let (tx, rx) = mpsc::channel::<Result<IndexEvent>>(32);
    let indexer = self.indexer.clone();
    let embedder_cell = self.embedder.clone();
    let chunker_cell = self.chunker.clone();
    let hub = self.hub.clone();
    let config = self.config.clone();
    let storage = self.storage.clone();
    let verbose = self.verbose;

    tokio::spawn(async move {
      let result: Result<()> = async {
        let indexer = match indexer {
          Some(indexer) => indexer,
          None => {
            let mut emit = |spec: &ModelSpec, phase: DownloadPhase| {
              let event = match phase {
                DownloadPhase::Start => IndexEvent::ModelDownloadStart {
                  role: spec.role,
                  repo_id: spec.repo_id.clone(),
                  filename: spec.filename.clone(),
                },
                DownloadPhase::Complete { elapsed_ms } => IndexEvent::ModelDownloadComplete {
                  role: spec.role,
                  elapsed_ms,
                },
              };
              let _ = tx.try_send(Ok(event));
            };
            {
              let _step = verbose.start("load embedding model");
              Self::ensure_embedder(&embedder_cell, &hub, &config, storage.as_ref(), &mut emit).await?;
            }
            let chunker = {
              let _step = verbose.start("load tokenizer");
              Self::ensure_chunker(&chunker_cell, &hub, &config, storage.as_ref(), &mut emit).await?
            };
            Arc::new(Indexer::new(IndexerConfig {
              chunker,
              embedder: embedder_cell,
              segmenter: Arc::new(JiebaSegmenter),
              storage,
              readers: Self::default_readers(),
              verbose,
              embedding_spec: config.models.embedding.to_spec(ModelRole::Embedding),
              chunk_size: config.indexing.chunk_size,
              chunk_overlap: config.indexing.chunk_overlap,
            }))
          }
        };
        let mut stream = std::pin::pin!(indexer.index_sources_stream(sources));
        while let Some(event) = stream.next().await {
          if tx.send(event).await.is_err() {
            return Ok(()); // consumer dropped
          }
        }
        Ok(())
      }
      .await;
      if let Err(e) = result {
        let _ = tx.send(Err(e)).await;
      }
    });

    ReceiverStream::new(rx)
  }

  /// Lazily load the embedding model into the shared cell. `on_download`
  /// fires once if the model file actually hits the network (cache hits are
  /// silent). After this returns, `storage` has been initialized with the
  /// embedding dimension.
  async fn ensure_embedder(
    embedder_cell: &Arc<OnceCell<Arc<dyn Embedder>>>,
    hub: &ModelHub,
    config: &SemqConfig,
    storage: &dyn Storage,
    mut on_download: impl FnMut(&ModelSpec, DownloadPhase),
  ) -> Result<Arc<dyn Embedder>> {
    embedder_cell
      .get_or_try_init(|| async {
        let emb_spec = config.models.embedding.to_spec(ModelRole::Embedding);
        Self::ensure_model_file(hub, storage, &emb_spec, &mut on_download).await?;
        let embedder: Arc<dyn Embedder> = Arc::new(FastEmbedEmbedder::from_model_hub(hub, &emb_spec).await?);
        storage.init(embedder.dimension())?;
        Ok(embedder)
      })
      .await
      .map(Arc::clone)
  }

  /// Lazily load the tokenizer-backed chunker. Only indexing needs it —
  /// search/ask must not pay the tokenizer download.
  async fn ensure_chunker(
    chunker_cell: &Arc<OnceCell<Arc<dyn Chunker>>>,
    hub: &ModelHub,
    config: &SemqConfig,
    storage: &dyn Storage,
    mut on_download: impl FnMut(&ModelSpec, DownloadPhase),
  ) -> Result<Arc<dyn Chunker>> {
    chunker_cell
      .get_or_try_init(|| async {
        let emb_spec = config.models.embedding.to_spec(ModelRole::Embedding);
        let tokenizer_spec = ModelSpec {
          role: ModelRole::Tokenizer,
          repo_id: emb_spec.repo_id.clone(),
          filename: config.models.embedding.tokenizer_filename.clone(),
          revision: emb_spec.revision.clone(),
          checksum: None,
        };
        Self::ensure_model_file(hub, storage, &tokenizer_spec, &mut on_download).await?;
        Self::build_chunker(
          hub,
          &emb_spec,
          &config.models.embedding.tokenizer_filename,
          &config.indexing,
        )
        .await
      })
      .await
      .map(Arc::clone)
  }

  /// Lazily load the reranker into the shared cell.
  async fn ensure_reranker(
    reranker_cell: &Arc<OnceCell<Arc<dyn Reranker>>>,
    hub: &ModelHub,
    config: &SemqConfig,
    storage: &dyn Storage,
    mut on_download: impl FnMut(&ModelSpec, DownloadPhase),
  ) -> Result<Arc<dyn Reranker>> {
    reranker_cell
      .get_or_try_init(|| async {
        let spec = config.models.reranker.to_spec(ModelRole::Reranker);
        Self::ensure_model_file(hub, storage, &spec, &mut on_download).await?;
        Ok(Arc::new(FastEmbedReranker::from_model_hub(hub, &spec).await?) as Arc<dyn Reranker>)
      })
      .await
      .map(Arc::clone)
  }

  /// Lazily load the chat LLM into the shared cell.
  async fn ensure_llm(
    llm_cell: &Arc<OnceCell<Arc<dyn Llm>>>,
    hub: &ModelHub,
    config: &SemqConfig,
    storage: &dyn Storage,
    mut on_download: impl FnMut(&ModelSpec, DownloadPhase),
  ) -> Result<Arc<dyn Llm>> {
    llm_cell
      .get_or_try_init(|| async {
        let spec = config.models.llm.to_spec(ModelRole::Chat);
        let llm_config: LlmConfig = config.llm.clone().try_into()?;
        Self::ensure_model_file(hub, storage, &spec, &mut on_download).await?;
        Ok(Arc::new(GgufLlm::from_model_hub(hub, &spec, &llm_config).await?) as Arc<dyn Llm>)
      })
      .await
      .map(Arc::clone)
  }

  /// Build a retriever over the shared lazy cells for engines opened with
  /// `Engine::open` (no pre-built retriever).
  fn lazy_retriever(
    storage: Arc<dyn Storage>,
    embedder_cell: Arc<OnceCell<Arc<dyn Embedder>>>,
    reranker_cell: Arc<OnceCell<Arc<dyn Reranker>>>,
    retrieval: &RetrievalConfig,
    verbose: Verbose,
  ) -> Retriever {
    Retriever::new(RetrieverConfig {
      storage,
      embedder: embedder_cell,
      segmenter: Arc::new(JiebaSegmenter),
      reranker: Some(reranker_cell),
      bm25_top_k: retrieval.bm25_top_k,
      vector_top_k: retrieval.vector_top_k,
      rrf_k: retrieval.rrf_k,
      rerank_top_n: retrieval.rerank_top_n,
      verbose,
    })
  }

  /// Resolve a model file, emitting download events only when the file is not
  /// already in the local cache. Records the model version either way.
  async fn ensure_model_file(
    hub: &ModelHub,
    storage: &dyn Storage,
    spec: &ModelSpec,
    on_download: &mut impl FnMut(&ModelSpec, DownloadPhase),
  ) -> Result<()> {
    if hub.is_cached(spec) {
      return hub.ensure(spec, storage).await.map(|_| ());
    }
    on_download(spec, DownloadPhase::Start);
    let start = Instant::now();
    hub.ensure(spec, storage).await?;
    on_download(
      spec,
      DownloadPhase::Complete {
        elapsed_ms: start.elapsed().as_millis() as u64,
      },
    );
    Ok(())
  }

  /// Search for passages. Drives the same lazy-loading stream as
  /// [`Engine::search_stream`] and returns the final hit list.
  pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>> {
    let mut stream = std::pin::pin!(self.spawn_search(query.to_string(), top_k));
    let mut hits = Vec::new();
    while let Some(event) = stream.next().await {
      if let SearchEvent::Completed { hits: final_hits, .. } = event? {
        hits = final_hits;
      }
    }
    Ok(hits)
  }

  /// Streaming variant of [`Engine::search`]: model download events (first
  /// use, uncached) followed by retrieval stage events, terminated by
  /// `Completed`.
  pub fn search_stream(
    &self,
    query: impl Into<String>,
    top_k: usize,
  ) -> Result<impl Stream<Item = Result<SearchEvent>> + Send + 'static> {
    Ok(self.spawn_search(query.into(), top_k))
  }

  /// Spawn a search pipeline. Engines built by `Engine::new` reuse their
  /// pre-built retriever; engines from `Engine::open` lazily load the
  /// embedder + reranker (emitting download events when a file actually hits
  /// the network) and construct a retriever on first use.
  fn spawn_search(&self, query: String, top_k: usize) -> impl Stream<Item = Result<SearchEvent>> + Send + 'static {
    let (tx, rx) = mpsc::channel::<Result<SearchEvent>>(32);
    let retriever = self.retriever.clone();
    let embedder_cell = self.embedder.clone();
    let reranker_cell = self.reranker.clone();
    let hub = self.hub.clone();
    let config = self.config.clone();
    let storage = self.storage.clone();
    let verbose = self.verbose;

    tokio::spawn(async move {
      let result: Result<()> = async {
        let retriever = match retriever {
          Some(retriever) => retriever,
          None => {
            let mut emit = |spec: &ModelSpec, phase: DownloadPhase| {
              let event = match phase {
                DownloadPhase::Start => SearchEvent::ModelDownloadStart {
                  role: spec.role,
                  repo_id: spec.repo_id.clone(),
                  filename: spec.filename.clone(),
                },
                DownloadPhase::Complete { elapsed_ms } => SearchEvent::ModelDownloadComplete {
                  role: spec.role,
                  elapsed_ms,
                },
              };
              let _ = tx.try_send(Ok(event));
            };
            {
              let _step = verbose.start("load embedding model");
              Self::ensure_embedder(&embedder_cell, &hub, &config, storage.as_ref(), &mut emit).await?;
            }
            {
              let _step = verbose.start("load reranker model");
              Self::ensure_reranker(&reranker_cell, &hub, &config, storage.as_ref(), &mut emit).await?;
            }
            Arc::new(Self::lazy_retriever(
              storage,
              embedder_cell,
              reranker_cell,
              &config.retrieval,
              verbose,
            ))
          }
        };
        let mut stream = std::pin::pin!(retriever.search_stream(query, top_k));
        while let Some(event) = stream.next().await {
          if tx.send(event).await.is_err() {
            return Ok(()); // consumer dropped
          }
        }
        Ok(())
      }
      .await;
      if let Err(e) = result {
        let _ = tx.send(Err(e)).await;
      }
    });

    ReceiverStream::new(rx)
  }

  /// Ask a question and get a cited answer. Drives the same lazy-loading
  /// stream as [`Engine::ask_stream`] and returns the terminal answer.
  pub async fn ask(&self, query: &str) -> Result<semquery_core::Answer> {
    let mut stream = std::pin::pin!(self.spawn_ask(query.to_string()));
    let mut answer = None;
    while let Some(event) = stream.next().await {
      if let semquery_core::AskEvent::AnswerComplete { answer: a, .. } = event? {
        answer = Some(a);
      }
    }
    answer.ok_or_else(|| semquery_core::SynthError::Other("ask stream ended without AnswerComplete".into()).into())
  }

  /// Streaming variant of [`Engine::ask`]: model download events (first use,
  /// uncached), then retrieval stage events, then each LLM token as it is
  /// decoded, terminated by `AnswerComplete`.
  pub fn ask_stream(
    &self,
    query: impl Into<String>,
  ) -> Result<impl Stream<Item = std::result::Result<semquery_core::AskEvent, semquery_core::SemqError>> + Send + 'static>
  {
    Ok(self.spawn_ask(query.into()))
  }

  /// Spawn an ask pipeline. Engines built by `Engine::new` reuse their
  /// pre-built synthesizer; engines from `Engine::open` lazily load
  /// embedder + reranker + LLM (emitting download events when a file
  /// actually hits the network) and construct the retriever + synthesizer on
  /// first use.
  fn spawn_ask(
    &self,
    query: String,
  ) -> impl Stream<Item = std::result::Result<semquery_core::AskEvent, semquery_core::SemqError>> + Send + 'static {
    let (tx, rx) = mpsc::channel::<std::result::Result<semquery_core::AskEvent, semquery_core::SemqError>>(32);
    let synthesizer = self.synthesizer.clone();
    let retriever = self.retriever.clone();
    let embedder_cell = self.embedder.clone();
    let reranker_cell = self.reranker.clone();
    let llm_cell = self.llm.clone();
    let hub = self.hub.clone();
    let config = self.config.clone();
    let storage = self.storage.clone();
    let verbose = self.verbose;

    tokio::spawn(async move {
      let result: Result<()> = async {
        let synthesizer = match synthesizer {
          Some(synthesizer) => synthesizer,
          None => {
            let mut emit = |spec: &ModelSpec, phase: DownloadPhase| {
              let event = match phase {
                DownloadPhase::Start => semquery_core::AskEvent::ModelDownloadStart {
                  role: spec.role,
                  repo_id: spec.repo_id.clone(),
                  filename: spec.filename.clone(),
                },
                DownloadPhase::Complete { elapsed_ms } => semquery_core::AskEvent::ModelDownloadComplete {
                  role: spec.role,
                  elapsed_ms,
                },
              };
              let _ = tx.try_send(Ok(event));
            };
            {
              let _step = verbose.start("load embedding model");
              Self::ensure_embedder(&embedder_cell, &hub, &config, storage.as_ref(), &mut emit).await?;
            }
            {
              let _step = verbose.start("load reranker model");
              Self::ensure_reranker(&reranker_cell, &hub, &config, storage.as_ref(), &mut emit).await?;
            }
            {
              let _step = verbose.start("load LLM");
              Self::ensure_llm(&llm_cell, &hub, &config, storage.as_ref(), &mut emit).await?;
            }
            let retriever = match retriever {
              Some(retriever) => retriever,
              None => Arc::new(Self::lazy_retriever(
                storage,
                embedder_cell,
                reranker_cell,
                &config.retrieval,
                verbose,
              )),
            };
            Synthesizer::new(SynthesizerConfig {
              retriever,
              llm: llm_cell,
              verbose,
            })
          }
        };
        let mut stream = std::pin::pin!(synthesizer.ask_stream(query));
        while let Some(event) = stream.next().await {
          if tx.send(event).await.is_err() {
            return Ok(()); // consumer dropped
          }
        }
        Ok(())
      }
      .await;
      if let Err(e) = result {
        let _ = tx.send(Err(e)).await;
      }
    });

    ReceiverStream::new(rx)
  }

  pub fn status(&self) -> Result<EngineStatus> {
    let docs = self.storage.list_documents()?;
    let collections = self.storage.list_collections()?;
    let chunks = self.storage.count_chunks()?;
    Ok(EngineStatus {
      documents: docs.len(),
      chunks,
      collections,
    })
  }
}

#[cfg(test)]
mod tests {
  fn test_readers() -> ReaderRegistry {
    let mut reg = ReaderRegistry::new();
    reg.register(Arc::new(TextFileReader::new()));
    #[cfg(feature = "pdf")]
    reg.register(Arc::new(PdfReader::new()));
    #[cfg(feature = "docx")]
    reg.register(Arc::new(DocxReader::new()));
    reg
  }

  use super::*;
  use semquery_core::{ChunkCandidate, Chunker, Embedder, Llm, Storage};
  use tempfile::TempDir;
  use tokio_stream::StreamExt;

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

  struct StubLlm;

  #[async_trait::async_trait]
  impl Llm for StubLlm {
    async fn complete(&self, _prompt: &str) -> Result<String> {
      Ok("This is a stub answer [1].".to_string())
    }
  }

  fn test_storage(tmp: &TempDir) -> Arc<dyn Storage> {
    let storage = Arc::new(SqliteStorage::open(tmp.path().join("test.db")).unwrap()) as Arc<dyn Storage>;
    storage.init(512).unwrap();
    storage
  }

  fn test_components(storage: Arc<dyn Storage>) -> EngineComponents {
    EngineComponents {
      storage,
      hub: ModelHub::new(std::env::temp_dir().join("semq-test-model-cache")),
      config: crate::config::SemqConfig::default(),
      chunker: Arc::new(StubChunker),
      embedder: Arc::new(StubEmbedder { dim: 512 }),
      segmenter: Arc::new(JiebaSegmenter),
      reranker: None,
      llm: Some(Arc::new(StubLlm)),
      readers: test_readers(),
      retrieval: crate::config::RetrievalConfig {
        bm25_top_k: 100,
        vector_top_k: 100,
        rrf_k: 60,
        rerank_top_n: 20,
      },
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
    }
  }

  #[tokio::test]
  async fn test_engine_add_collection_and_status() {
    let tmp = TempDir::new().unwrap();
    let storage = test_storage(&tmp);
    let engine = Engine::new(test_components(storage));

    let notes_dir = TempDir::new().unwrap();
    engine.add_collection("notes", notes_dir.path()).unwrap();

    let collections = engine.list_collections().unwrap();
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0].name, "notes");

    let status = engine.status().unwrap();
    assert_eq!(status.collections.len(), 1);
    assert_eq!(status.documents, 0);
  }

  #[tokio::test]
  async fn test_engine_index_and_search() {
    let tmp = TempDir::new().unwrap();
    let storage = test_storage(&tmp);
    let engine = Engine::new(test_components(storage));

    let notes_dir = TempDir::new().unwrap();
    std::fs::write(notes_dir.path().join("note.txt"), "今天是我的生日").unwrap();
    engine.add_collection("notes", notes_dir.path()).unwrap();

    let stats = engine.index().await.unwrap();
    assert!(stats.chunks_indexed > 0);

    let hits = engine.search("生日", 5).await.unwrap();
    assert!(!hits.is_empty());
    assert!(hits[0].chunk.text.contains("生日"));
  }

  #[tokio::test]
  async fn test_engine_search_stream() {
    let tmp = TempDir::new().unwrap();
    let storage = test_storage(&tmp);
    let engine = Engine::new(test_components(storage));

    let notes_dir = TempDir::new().unwrap();
    std::fs::write(notes_dir.path().join("note.txt"), "今天是我的生日").unwrap();
    engine.add_collection("notes", notes_dir.path()).unwrap();

    let stats = engine.index().await.unwrap();
    assert!(stats.chunks_indexed > 0);

    let mut stream = engine.search_stream("生日", 5).unwrap();
    let mut completed_hits = Vec::new();
    while let Some(event) = stream.next().await {
      match event.unwrap() {
        SearchEvent::Completed { hits, .. } => completed_hits = hits,
        _ => {}
      }
    }
    assert!(!completed_hits.is_empty());
    assert!(completed_hits[0].chunk.text.contains("生日"));
  }

  #[tokio::test]
  async fn test_engine_index_stream() {
    let tmp = TempDir::new().unwrap();
    let storage = test_storage(&tmp);
    let engine = Engine::new(test_components(storage));

    let notes_dir = TempDir::new().unwrap();
    std::fs::write(notes_dir.path().join("note.txt"), "今天是我的生日").unwrap();
    engine.add_collection("notes", notes_dir.path()).unwrap();

    let mut stream = engine.index_stream().unwrap();
    let mut completed = None;
    let mut source_complete = None;
    while let Some(event) = stream.next().await {
      match event.unwrap() {
        IndexEvent::SourceComplete { source_id, chunks } => source_complete = Some((source_id, chunks)),
        IndexEvent::Complete { files, chunks, .. } => completed = Some((files, chunks)),
        _ => {}
      }
    }
    assert!(completed.is_some());
    let (files, chunks) = completed.unwrap();
    assert_eq!(files, 1);
    assert!(chunks > 0);
    assert_eq!(source_complete.map(|(id, _)| id), Some("notes".to_string()));
  }

  #[tokio::test]
  async fn test_engine_index_one_stream() {
    let tmp = TempDir::new().unwrap();
    let storage = test_storage(&tmp);
    let engine = Engine::new(test_components(storage));

    let notes_dir = TempDir::new().unwrap();
    std::fs::write(notes_dir.path().join("note.txt"), "定价方案选坐席制").unwrap();
    engine.add_collection("notes", notes_dir.path()).unwrap();

    let mut stream = engine.index_one_stream("notes").unwrap();
    let mut completed = None;
    while let Some(event) = stream.next().await {
      if let IndexEvent::Complete { files, chunks, .. } = event.unwrap() {
        completed = Some((files, chunks));
      }
    }
    assert!(completed.is_some());
    let (files, chunks) = completed.unwrap();
    assert_eq!(files, 1);
    assert!(chunks > 0);
  }

  #[tokio::test]
  async fn test_engine_ask() {
    let tmp = TempDir::new().unwrap();
    let storage = test_storage(&tmp);
    let engine = Engine::new(test_components(storage));

    let notes_dir = TempDir::new().unwrap();
    std::fs::write(notes_dir.path().join("note.txt"), "定价方案选坐席制").unwrap();
    engine.add_collection("notes", notes_dir.path()).unwrap();
    engine.index().await.unwrap();

    let answer = engine.ask("定价方案").await.unwrap();
    assert!(!answer.text.is_empty());
  }

  #[tokio::test]
  async fn test_engine_open_loads_no_models() {
    let tmp = TempDir::new().unwrap();
    // Point the embedding model at a repo that cannot exist so the lazy load
    // triggered by index() fails fast (404 / DNS error) instead of
    // downloading a real model in a unit test.
    let mut semq_config = crate::config::SemqConfig::default();
    semq_config.models.embedding.repo_id = "nonexistent/repo".into();
    let config = EngineConfig {
      workspace_path: tmp.path().to_path_buf(),
      model_cache_dir: tmp.path().join("models"),
      config: semq_config,
      verbose: Verbose(false),
    };
    let engine = Engine::open(config).unwrap();

    // Storage-level operations work without any model.
    let notes_dir = TempDir::new().unwrap();
    engine.add_collection("notes", notes_dir.path()).unwrap();
    let status = engine.status().unwrap();
    assert_eq!(status.collections.len(), 1);
    assert_eq!(status.collections[0].name, "notes");

    // index() drives lazy loading on an `open` engine: the stream emits a
    // ModelDownloadStart event and then fails because the model is unfetchable.
    let mut stream = std::pin::pin!(engine.index_stream().unwrap());
    let mut saw_download_start = false;
    let mut failed = false;
    while let Some(event) = stream.next().await {
      match event {
        Ok(IndexEvent::ModelDownloadStart { role, .. }) => {
          assert_eq!(role, semquery_core::ModelRole::Embedding);
          saw_download_start = true;
        }
        Ok(_) => {}
        Err(_) => {
          failed = true;
          break;
        }
      }
    }
    assert!(saw_download_start, "lazy index must emit a download-start event first");
    assert!(
      failed,
      "lazy index must fail when the embedding model cannot be fetched"
    );

    // Retrieval / ask are lazy too: they attempt to load the (unfetchable)
    // embedding model and fail inside the stream.
    assert!(engine.search("生日", 5).await.is_err());
    let mut search_stream = std::pin::pin!(engine.search_stream("生日", 5).unwrap());
    let mut search_failed = false;
    while let Some(event) = search_stream.next().await {
      if event.is_err() {
        search_failed = true;
        break;
      }
    }
    assert!(search_failed, "lazy search must fail when the model cannot be fetched");
    assert!(engine.ask("生日").await.is_err());
    let mut ask_stream = std::pin::pin!(engine.ask_stream("生日").unwrap());
    let mut ask_failed = false;
    while let Some(event) = ask_stream.next().await {
      if event.is_err() {
        ask_failed = true;
        break;
      }
    }
    assert!(ask_failed, "lazy ask must fail when the model cannot be fetched");
  }

  #[tokio::test]
  async fn test_open_engine_lazy_index_with_cached_models() {
    let tmp = TempDir::new().unwrap();
    let engine = Engine::open(EngineConfig {
      workspace_path: tmp.path().to_path_buf(),
      model_cache_dir: tmp.path().join("models"),
      config: crate::config::SemqConfig::default(),
      verbose: Verbose(false),
    })
    .unwrap();

    // Simulate models that are already available: pre-fill the lazy cells
    // (tests live inside the engine module, so private fields are reachable)
    // and initialize the vector table with the stub dimension.
    engine.storage.init(512).unwrap();
    assert!(engine.embedder.set(Arc::new(StubEmbedder { dim: 512 }) as Arc<dyn Embedder>).is_ok());
    assert!(engine.chunker.set(Arc::new(StubChunker)).is_ok());

    let notes_dir = TempDir::new().unwrap();
    std::fs::write(notes_dir.path().join("note.txt"), "今天是我的生日").unwrap();
    engine.add_collection("notes", notes_dir.path()).unwrap();

    let mut stream = std::pin::pin!(engine.index_stream().unwrap());
    let mut completed = None;
    let mut download_events = 0;
    while let Some(event) = stream.next().await {
      match event.unwrap() {
        IndexEvent::Complete { files, chunks, .. } => completed = Some((files, chunks)),
        IndexEvent::ModelDownloadStart { .. } | IndexEvent::ModelDownloadComplete { .. } => download_events += 1,
        _ => {}
      }
    }

    let (files, chunks) = completed.expect("lazy index must complete");
    assert_eq!(files, 1);
    assert!(chunks > 0);
    assert_eq!(download_events, 0, "cached models must not emit download events");
  }

  #[tokio::test]
  async fn test_data_only_then_reopen_with_models() {
    let tmp = TempDir::new().unwrap();

    // Phase 1: data-only open — manages collections without any model.
    {
      let engine = Engine::open(EngineConfig {
        workspace_path: tmp.path().to_path_buf(),
        model_cache_dir: tmp.path().join("models"),
        config: crate::config::SemqConfig::default(),
        verbose: Verbose(false),
      })
      .unwrap();
      let notes_dir = TempDir::new().unwrap();
      engine.add_collection("notes", notes_dir.path()).unwrap();
      assert_eq!(engine.status().unwrap().collections.len(), 1);
    }

    // Phase 2: reopen the same workspace for model-backed use. This mirrors
    // what lazy loading does once the embedding dimension is known: open
    // storage + init(dimension). It must not hit SchemaMismatch from the
    // earlier init(0), and data written in phase 1 must survive.
    let storage = SqliteStorage::open_workspace(tmp.path()).unwrap();
    storage.init(512).unwrap();
    let collections = storage.list_collections().unwrap();
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0].name, "notes");
  }

  #[tokio::test]
  async fn test_index_then_data_only_reads_indexed_data() {
    let tmp = TempDir::new().unwrap();

    // Phase 1: full (model-backed) engine indexes a document into the workspace.
    {
      let storage: Arc<dyn Storage> = Arc::new(SqliteStorage::open_workspace(tmp.path()).unwrap());
      storage.init(512).unwrap();
      let engine = Engine::new(test_components(storage));

      let notes_dir = TempDir::new().unwrap();
      std::fs::write(notes_dir.path().join("note.txt"), "今天是我的生日").unwrap();
      engine.add_collection("notes", notes_dir.path()).unwrap();
      let stats = engine.index().await.unwrap();
      assert!(stats.chunks_indexed > 0);
    }

    // Phase 2: reopen via the canonical `open` — indexed data must be intact and readable.
    {
      let engine = Engine::open(EngineConfig {
        workspace_path: tmp.path().to_path_buf(),
        model_cache_dir: tmp.path().join("models"),
        config: crate::config::SemqConfig::default(),
        verbose: Verbose(false),
      })
      .unwrap();

      let status = engine.status().unwrap();
      assert_eq!(status.documents, 1);
      assert!(status.chunks > 0);
      assert_eq!(status.collections.len(), 1);
    }

    // Phase 3: the document itself is still retrievable at the storage level.
    let storage = SqliteStorage::open_workspace(tmp.path()).unwrap();
    let docs = storage.list_documents().unwrap();
    assert_eq!(docs.len(), 1);
    assert!(storage.get_document(&docs[0].id).unwrap().is_some());
  }

  #[tokio::test]
  async fn test_engine_ask_without_llm_errors() {
    let tmp = TempDir::new().unwrap();
    let storage = test_storage(&tmp);
    // No LLM injected: ask() falls back to lazy loading, which must fail
    // fast. Point reranker/llm at repos that cannot exist so no real
    // download is attempted (the embedder cell is pre-filled with a stub).
    let mut semq_config = crate::config::SemqConfig::default();
    semq_config.models.reranker.repo_id = "nonexistent/repo".into();
    semq_config.models.llm.repo_id = "nonexistent/repo".into();
    let components = EngineComponents {
      storage,
      hub: ModelHub::new(std::env::temp_dir().join("semq-test-model-cache")),
      config: semq_config,
      chunker: Arc::new(StubChunker),
      embedder: Arc::new(StubEmbedder { dim: 512 }),
      segmenter: Arc::new(JiebaSegmenter),
      reranker: None,
      llm: None,
      readers: test_readers(),
      retrieval: crate::config::RetrievalConfig {
        bm25_top_k: 100,
        vector_top_k: 100,
        rrf_k: 60,
        rerank_top_n: 20,
      },
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
    };
    let engine = Engine::new(components);
    let result = engine.ask("test").await;
    assert!(result.is_err());
  }
}
