//! Engine facade — assembles all components into a single API.

use std::path::{Path, PathBuf};
use std::sync::Arc;

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
use tokio_stream::Stream;

use crate::config::{DocqConfig, RetrievalConfig};
use semquery_storage::SqliteStorage;
use semquery_synth::{Synthesizer, SynthesizerConfig};

pub struct EngineConfig {
  pub workspace_path: PathBuf,
  pub model_cache_dir: PathBuf,
  pub config: DocqConfig,
  pub verbose: Verbose,
}

/// Pre-built components for `Engine::new` (dependency injection).
/// Tests construct this with stub implementations; `Engine::open_for_*`
/// constructs it with real model backends loaded on demand.
pub struct EngineComponents {
  pub storage: Arc<dyn Storage>,
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
  indexer: Indexer,
  retriever: Arc<Retriever>,
  synthesizer: Option<Synthesizer>,
  verbose: Verbose,
}

impl Engine {
  pub fn new(components: EngineComponents) -> Self {
    let EngineComponents {
      storage,
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

    let indexer = Indexer::new(IndexerConfig {
      chunker,
      embedder: embedder.clone(),
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
      embedder,
      segmenter,
      reranker,
      bm25_top_k: retrieval.bm25_top_k,
      vector_top_k: retrieval.vector_top_k,
      rrf_k: retrieval.rrf_k,
      rerank_top_n: retrieval.rerank_top_n,
      verbose,
    }));

    let synthesizer = llm.map(|llm| {
      Synthesizer::new(SynthesizerConfig {
        retriever: retriever.clone(),
        llm,
        verbose,
      })
    });

    Self {
      storage,
      indexer,
      retriever,
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

  async fn load_embedding(
    hub: &ModelHub,
    storage: &dyn Storage,
    spec: &ModelSpec,
    tokenizer_filename: &str,
    indexing: &crate::config::IndexingConfig,
  ) -> Result<(Arc<dyn Embedder>, Arc<dyn Chunker>)> {
    hub.ensure(spec, storage).await?;
    let embedder = Arc::new(FastEmbedEmbedder::from_model_hub(hub, spec).await?);
    let chunker = Self::build_chunker(hub, spec, tokenizer_filename, indexing).await?;
    Ok((embedder, chunker))
  }

  async fn load_reranker(hub: &ModelHub, storage: &dyn Storage, spec: &ModelSpec) -> Result<Arc<dyn Reranker>> {
    hub.ensure(spec, storage).await?;
    Ok(Arc::new(FastEmbedReranker::from_model_hub(hub, spec).await?))
  }

  fn load_reranker_sync(
    hub: ModelHub,
    storage: Arc<dyn Storage>,
    spec: ModelSpec,
    verbose: Verbose,
  ) -> Result<Arc<dyn Reranker>> {
    let _step = verbose.start("load reranker model");
    hub.ensure_sync(&spec, storage.as_ref())?;
    Ok(Arc::new(FastEmbedReranker::from_model_hub_sync(&hub, &spec)?))
  }

  fn load_llm_sync(
    hub: ModelHub,
    storage: Arc<dyn Storage>,
    spec: ModelSpec,
    llm_config: LlmConfig,
    verbose: Verbose,
  ) -> Result<Arc<dyn Llm>> {
    let _step = verbose.start("load LLM");
    hub.ensure_sync(&spec, storage.as_ref())?;
    Ok(Arc::new(GgufLlm::from_model_hub_sync(&hub, &spec, &llm_config)?))
  }

  // ---- On-demand open methods ----

  /// Open for indexing: loads embedding model only (~100 MB).
  pub async fn open_for_index(config: EngineConfig) -> Result<Self> {
    let (components, _) = Self::build_index_components(&config).await?;
    Ok(Self::new(components))
  }

  /// Open for search: loads embedding + reranker models (~1.1 GB).
  pub async fn open_for_search(config: EngineConfig) -> Result<Self> {
    let (components, _) = Self::build_search_components(&config).await?;
    Ok(Self::new(components))
  }

  /// Open for ask: loads all models (~6 GB).
  pub async fn open_for_ask(config: EngineConfig) -> Result<Self> {
    let (components, _) = Self::build_ask_components(&config).await?;
    Ok(Self::new(components))
  }

  async fn build_index_components(engine_config: &EngineConfig) -> Result<(EngineComponents, ModelHub)> {
    let hub = ModelHub::new(engine_config.model_cache_dir.clone());
    let storage = Self::open_storage(&engine_config.workspace_path)?;
    let emb_spec = engine_config.config.models.embedding.to_spec(ModelRole::Embedding);
    let tokenizer_filename = engine_config.config.models.embedding.tokenizer_filename.clone();
    let (embedder, chunker) = {
      let _step = engine_config.verbose.start("load embedding model");
      Self::load_embedding(
        &hub,
        storage.as_ref(),
        &emb_spec,
        &tokenizer_filename,
        &engine_config.config.indexing,
      )
      .await?
    };
    storage.init(embedder.dimension())?;

    let components = EngineComponents {
      storage,
      chunker,
      embedder,
      segmenter: Arc::new(JiebaSegmenter),
      reranker: None,
      llm: None,
      readers: Self::default_readers(),
      retrieval: engine_config.config.retrieval.clone(),
      verbose: engine_config.verbose,
      embedding_spec: emb_spec,
      chunk_size: engine_config.config.indexing.chunk_size,
      chunk_overlap: engine_config.config.indexing.chunk_overlap,
    };

    Ok((components, hub))
  }

  async fn build_search_components(engine_config: &EngineConfig) -> Result<(EngineComponents, ModelHub)> {
    let (mut components, hub) = Self::build_index_components(engine_config).await?;
    let rerank_spec = engine_config.config.models.reranker.to_spec(ModelRole::Reranker);
    components.reranker = Some({
      let _step = engine_config.verbose.start("load reranker model");
      Self::load_reranker(&hub, components.storage.as_ref(), &rerank_spec).await?
    });

    Ok((components, hub))
  }

  async fn build_ask_components(engine_config: &EngineConfig) -> Result<(EngineComponents, ModelHub)> {
    let (mut components, hub) = Self::build_index_components(engine_config).await?;

    let rerank_spec = engine_config.config.models.reranker.to_spec(ModelRole::Reranker);
    let llm_spec = engine_config.config.models.llm.to_spec(ModelRole::Chat);
    let llm_config: LlmConfig = engine_config.config.llm.clone().try_into()?;
    let storage = components.storage.clone();
    let verbose = engine_config.verbose;

    // Reranker and LLM are independent; load them in parallel
    // on blocking threads so their downloads overlap.
    let rerank_hub = hub.clone();
    let llm_hub = hub.clone();
    let rerank_storage = storage.clone();
    let llm_storage = storage.clone();
    let rerank_verbose = verbose;
    let llm_verbose = verbose;

    let (reranker, llm) = tokio::join!(
      tokio::task::spawn_blocking(move || {
        Self::load_reranker_sync(rerank_hub, rerank_storage, rerank_spec, rerank_verbose)
      }),
      tokio::task::spawn_blocking(move || {
        Self::load_llm_sync(llm_hub, llm_storage, llm_spec, llm_config, llm_verbose)
      }),
    );

    components.reranker = Some(reranker.map_err(|e| semquery_core::ModelError::TaskJoin(e.to_string()))??);
    components.llm = Some(llm.map_err(|e| semquery_core::ModelError::TaskJoin(e.to_string()))??);

    Ok((components, hub))
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
    let indexer = self.indexer.clone();
    Ok(indexer.index_sources_stream(sources))
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
    let indexer = self.indexer.clone();
    Ok(indexer.index_sources_stream(vec![(col.name, col.path)]))
  }

  pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>> {
    self.retriever.search(query, top_k).await
  }

  pub fn search_stream(
    &self,
    query: impl Into<String>,
    top_k: usize,
  ) -> impl Stream<Item = Result<SearchEvent>> + Send + 'static {
    self.retriever.clone().search_stream(query, top_k)
  }

  pub async fn ask(&self, query: &str) -> Result<semquery_core::Answer> {
    let synth = self.synthesizer.as_ref().ok_or(semquery_core::LlmError::NotLoaded)?;
    synth.ask(query).await
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

    let mut stream = engine.search_stream("生日", 5);
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
  async fn test_engine_ask_without_llm_errors() {
    let tmp = TempDir::new().unwrap();
    let storage = test_storage(&tmp);
    let components = EngineComponents {
      storage,
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
