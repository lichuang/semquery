use std::sync::Mutex;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use semquery_core::{EmbedError, Embedder, ModelError, ModelSpec, Result};

use crate::{BGE_SMALL_ZH_V1_5_REPO, EMBEDDING_REPO_BGE_LARGE_ZH, EMBEDDING_REPO_BGE_M3, ModelHub};

pub struct FastEmbedEmbedder {
  inner: Mutex<TextEmbedding>,
  model_name: String,
  dim: usize,
}

impl FastEmbedEmbedder {
  pub async fn from_model_hub(hub: &ModelHub, spec: &ModelSpec) -> Result<Self> {
    let model = embedding_model_for(&spec.repo_id)?;
    let dim = TextEmbedding::get_model_info(&model).map_err(|e| ModelError::ModelInfoFailed(e.to_string()))?.dim;
    let options = TextInitOptions::new(model).with_cache_dir(hub.cache_dir().to_path_buf());
    let inner = TextEmbedding::try_new(options).map_err(|e| ModelError::ModelInitFailed(e.to_string()))?;
    Ok(Self {
      inner: Mutex::new(inner),
      model_name: spec.repo_id.clone(),
      dim,
    })
  }
}

fn embedding_model_for(repo_id: &str) -> Result<EmbeddingModel> {
  match repo_id {
    BGE_SMALL_ZH_V1_5_REPO => Ok(EmbeddingModel::BGESmallZHV15),
    EMBEDDING_REPO_BGE_LARGE_ZH => Ok(EmbeddingModel::BGELargeZHV15),
    EMBEDDING_REPO_BGE_M3 => Ok(EmbeddingModel::BGEM3),
    other => Err(ModelError::UnsupportedModel(format!("unsupported embedding model: {other}")).into()),
  }
}

#[async_trait::async_trait]
impl Embedder for FastEmbedEmbedder {
  fn dimension(&self) -> usize {
    self.dim
  }

  fn model_name(&self) -> &str {
    &self.model_name
  }

  async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    let mut inner = self.inner.lock().map_err(|_| EmbedError::MutexPoisoned)?;
    let embeddings = inner.embed(texts, None).map_err(|e| EmbedError::InferenceFailed(e.to_string()))?;
    Ok(embeddings)
  }
}
