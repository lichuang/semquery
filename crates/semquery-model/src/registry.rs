use semquery_core::{ModelRole, ModelSpec};

// ---- Default embedding model: Xenova/bge-small-zh-v1.5 ----

pub const BGE_SMALL_ZH_V1_5_REPO: &str = "Xenova/bge-small-zh-v1.5";
pub const BGE_SMALL_ZH_V1_5_FILE: &str = "onnx/model.onnx";
/// Tokenizer file co-located in the same HF repo as the embedding model.
pub const BGE_SMALL_ZH_V1_5_TOKENIZER_FILE: &str = "tokenizer.json";
/// Maximum input length the embedding model accepts. Chunks exceeding this
/// are silently truncated by the ONNX runtime, so `SentenceSplitter` must
/// use this as `chunk_size` to avoid losing tail content.
pub const BGE_SMALL_ZH_V1_5_MAX_TOKENS: usize = 1024;
/// Output dimension of the default embedding vectors.
pub const BGE_SMALL_ZH_V1_5_DIMENSION: usize = 512;

// ---- Default reranker model: BAAI/bge-reranker-base ----

pub const BGE_RERANKER_BASE_REPO: &str = "BAAI/bge-reranker-base";
pub const BGE_RERANKER_BASE_FILE: &str = "onnx/model.onnx";

// ---- Default LLM model: Qwen/Qwen2.5-3B-Instruct-GGUF ----
// Qwen2.5-3B is a good default: small enough to load quickly on consumer
// hardware (~1.9 GB for q4_k_m) while still useful for RAG answers. The
// official repo provides an un-split q4_k_m file, so the single-filename
// loader works out of the box.
pub const QWEN2_5_3B_INSTRUCT_GGUF_REPO: &str = "Qwen/Qwen2.5-3B-Instruct-GGUF";
pub const QWEN2_5_3B_INSTRUCT_Q4_K_M_FILE: &str = "qwen2.5-3b-instruct-q4_k_m.gguf";

// ---- Other supported embedding repos (for repo_id → fastembed mapping) ----

pub const EMBEDDING_REPO_BGE_LARGE_ZH: &str = "Xenova/bge-large-zh-v1.5";
pub const EMBEDDING_REPO_BGE_M3: &str = "BAAI/bge-m3";

// ---- Other supported reranker repos (for repo_id → fastembed mapping) ----

pub const RERANKER_REPO_BGE_V2_M3: &str = "BAAI/bge-reranker-v2-m3";
/// Same model as above, mirrored under a different HF repo.
pub const RERANKER_REPO_BGE_V2_M3_ALT: &str = "rozgo/bge-reranker-v2-m3";
pub const RERANKER_REPO_JINA_V1_TURBO_EN: &str = "jinaai/jina-reranker-v1-turbo-en";
pub const RERANKER_JINA_V1_TURBO_EN_FILE: &str = "onnx/model.onnx";
pub const RERANKER_REPO_JINA_V2_MULTILINGUAL: &str = "jinaai/jina-reranker-v2-base-multilingual";

pub struct ModelRegistry;

impl ModelRegistry {
  pub fn default_embedding() -> ModelSpec {
    ModelSpec {
      role: ModelRole::Embedding,
      repo_id: BGE_SMALL_ZH_V1_5_REPO.into(),
      filename: BGE_SMALL_ZH_V1_5_FILE.into(),
      revision: "main".into(),
      checksum: None,
    }
  }

  pub fn default_reranker() -> ModelSpec {
    ModelSpec {
      role: ModelRole::Reranker,
      repo_id: RERANKER_REPO_JINA_V1_TURBO_EN.into(),
      filename: RERANKER_JINA_V1_TURBO_EN_FILE.into(),
      revision: "main".into(),
      checksum: None,
    }
  }

  pub fn default_llm() -> ModelSpec {
    ModelSpec {
      role: ModelRole::Chat,
      repo_id: QWEN2_5_3B_INSTRUCT_GGUF_REPO.into(),
      filename: QWEN2_5_3B_INSTRUCT_Q4_K_M_FILE.into(),
      revision: "main".into(),
      checksum: None,
    }
  }
}
