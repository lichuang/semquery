//! Global default configuration loaded from `config.toml`.
//!
//! The configuration file lives in the platform-specific default config
//! directory (e.g. `~/.config/semquery`) and controls which models to use, how
//! documents are chunked, and how retrieval / generation behave. It is loaded
//! independently of the workspace (data) directory.

use std::fs;
use std::path::{Path, PathBuf};

use semquery_core::{LlmConfig, LlmError, ModelRole, ModelSpec};
use serde::{Deserialize, Serialize};

pub const CONFIG_FILE_NAME: &str = "config.toml";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
  pub repo_id: String,
  pub filename: String,
  pub revision: String,
  #[serde(default = "default_tokenizer_filename")]
  pub tokenizer_filename: String,
}

fn default_tokenizer_filename() -> String {
  semquery_model::BGE_SMALL_ZH_V1_5_TOKENIZER_FILE.into()
}

impl ModelEntry {
  pub fn to_spec(&self, role: ModelRole) -> ModelSpec {
    ModelSpec {
      role,
      repo_id: self.repo_id.clone(),
      filename: self.filename.clone(),
      revision: self.revision.clone(),
      checksum: None,
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
  pub embedding: ModelEntry,
  pub reranker: ModelEntry,
  pub llm: ModelEntry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
  pub chunk_size: usize,
  pub chunk_overlap: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
  pub bm25_top_k: usize,
  pub vector_top_k: usize,
  pub rrf_k: usize,
  pub rerank_top_n: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmGenerationConfig {
  /// Stored as a string in TOML to avoid `f32` serialization artifacts
  /// (e.g. `0.7000000476837158`). Parsed when converting to `LlmConfig`.
  pub temperature: String,
  /// Stored as a string in TOML to avoid `f32` serialization artifacts.
  pub top_p: String,
  pub max_tokens: usize,
  pub n_ctx: u32,
  pub seed: u32,
  pub system_prompt: String,
}

impl TryFrom<LlmGenerationConfig> for LlmConfig {
  type Error = LlmError;

  fn try_from(c: LlmGenerationConfig) -> Result<Self, Self::Error> {
    Ok(Self {
      n_ctx: c.n_ctx,
      temperature: c.temperature.parse().map_err(|e| LlmError::InvalidConfig(format!("invalid temperature: {e}")))?,
      top_p: c.top_p.parse().map_err(|e| LlmError::InvalidConfig(format!("invalid top_p: {e}")))?,
      max_tokens: c.max_tokens,
      seed: c.seed,
      system_prompt: c.system_prompt,
    })
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
  /// Log level filter (e.g. "info", "debug", "warn"). Supports `RUST_LOG`-style
  /// target filtering like "semquery=debug,semquery_core=info".
  pub level: String,
  /// Optional log file path. If relative, resolved against the workspace.
  /// Defaults to `<workspace>/semquery.log` when omitted.
  pub file: Option<PathBuf>,
  /// Rotate the log file when it exceeds this size in megabytes.
  pub rotation_size_mb: usize,
  /// Maximum number of rotated log files to keep.
  pub max_files: usize,
  /// Also print log messages to the terminal while writing to file.
  pub duplicate_to_stderr: bool,
}

impl Default for LoggingConfig {
  fn default() -> Self {
    Self {
      level: "info".into(),
      file: None,
      rotation_size_mb: 10,
      max_files: 5,
      duplicate_to_stderr: false,
    }
  }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocqConfig {
  pub models: ModelsConfig,
  pub indexing: IndexingConfig,
  pub retrieval: RetrievalConfig,
  pub llm: LlmGenerationConfig,
  #[serde(default)]
  pub logging: LoggingConfig,
}

impl Default for ModelEntry {
  fn default() -> Self {
    Self {
      repo_id: semquery_model::BGE_SMALL_ZH_V1_5_REPO.into(),
      filename: semquery_model::BGE_SMALL_ZH_V1_5_FILE.into(),
      revision: "main".into(),
      tokenizer_filename: semquery_model::BGE_SMALL_ZH_V1_5_TOKENIZER_FILE.into(),
    }
  }
}

impl Default for ModelsConfig {
  fn default() -> Self {
    Self {
      embedding: ModelEntry {
        repo_id: semquery_model::BGE_SMALL_ZH_V1_5_REPO.into(),
        filename: semquery_model::BGE_SMALL_ZH_V1_5_FILE.into(),
        revision: "main".into(),
        tokenizer_filename: semquery_model::BGE_SMALL_ZH_V1_5_TOKENIZER_FILE.into(),
      },
      reranker: ModelEntry {
        repo_id: semquery_model::RERANKER_REPO_JINA_V1_TURBO_EN.into(),
        filename: semquery_model::RERANKER_JINA_V1_TURBO_EN_FILE.into(),
        revision: "main".into(),
        tokenizer_filename: default_tokenizer_filename(),
      },
      llm: ModelEntry {
        repo_id: semquery_model::QWEN2_5_3B_INSTRUCT_GGUF_REPO.into(),
        filename: semquery_model::QWEN2_5_3B_INSTRUCT_Q4_K_M_FILE.into(),
        revision: "main".into(),
        tokenizer_filename: default_tokenizer_filename(),
      },
    }
  }
}

impl Default for IndexingConfig {
  fn default() -> Self {
    Self {
      chunk_size: semquery_model::BGE_SMALL_ZH_V1_5_MAX_TOKENS,
      chunk_overlap: semquery_model::BGE_SMALL_ZH_V1_5_MAX_TOKENS / 10,
    }
  }
}

impl Default for RetrievalConfig {
  fn default() -> Self {
    Self {
      bm25_top_k: 100,
      vector_top_k: 100,
      rrf_k: 60,
      rerank_top_n: 20,
    }
  }
}

impl Default for LlmGenerationConfig {
  fn default() -> Self {
    Self {
      temperature: "0.7".into(),
      top_p: "0.9".into(),
      max_tokens: 512,
      n_ctx: 8192,
      seed: 0,
      system_prompt: "You are a helpful assistant.".into(),
    }
  }
}

impl DocqConfig {
  pub fn load(workspace: &Path) -> anyhow::Result<Self> {
    let path = Self::path(workspace);
    if !path.exists() {
      return Ok(Self::default());
    }
    Self::load_from_file(&path)
  }

  pub fn load_from_file(path: &Path) -> anyhow::Result<Self> {
    let text = fs::read_to_string(path).map_err(|e| anyhow::anyhow!("read config {}: {e}", path.display()))?;
    let config: Self = toml::from_str(&text).map_err(|e| anyhow::anyhow!("parse config {}: {e}", path.display()))?;
    Ok(config)
  }

  pub fn path(workspace: &Path) -> PathBuf {
    workspace.join(CONFIG_FILE_NAME)
  }

  pub fn to_toml(&self) -> anyhow::Result<String> {
    toml::to_string_pretty(self).map_err(|e| anyhow::anyhow!("serialize config: {e}"))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  #[test]
  fn test_default_config_roundtrip() {
    let config = DocqConfig::default();
    let toml = config.to_toml().unwrap();
    let parsed: DocqConfig = toml::from_str(&toml).unwrap();
    assert_eq!(parsed.indexing.chunk_size, config.indexing.chunk_size);
    assert_eq!(parsed.retrieval.rrf_k, config.retrieval.rrf_k);
    assert_eq!(parsed.llm.temperature, config.llm.temperature);
  }

  #[test]
  fn test_load_missing_returns_default() {
    let tmp = TempDir::new().unwrap();
    let config = DocqConfig::load(tmp.path()).unwrap();
    assert_eq!(config.indexing.chunk_size, semquery_model::BGE_SMALL_ZH_V1_5_MAX_TOKENS);
  }
}
