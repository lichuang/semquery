use std::sync::Mutex;

use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use semquery_core::{Chunk, ModelError, ModelSpec, Reranker, Result, RetrieveError, ScoredChunk};

use crate::{
  BGE_RERANKER_BASE_REPO, ModelHub, RERANKER_REPO_BGE_V2_M3, RERANKER_REPO_BGE_V2_M3_ALT,
  RERANKER_REPO_JINA_V1_TURBO_EN, RERANKER_REPO_JINA_V2_MULTILINGUAL,
};

pub struct FastEmbedReranker {
  inner: Mutex<TextRerank>,
  model_name: String,
}

impl FastEmbedReranker {
  pub async fn from_model_hub(hub: &ModelHub, spec: &ModelSpec) -> Result<Self> {
    let model = reranker_model_for(&spec.repo_id)?;
    let options = RerankInitOptions::new(model).with_cache_dir(hub.cache_dir().to_path_buf());
    let inner = TextRerank::try_new(options).map_err(|e| ModelError::ModelInitFailed(e.to_string()))?;
    Ok(Self {
      inner: Mutex::new(inner),
      model_name: spec.repo_id.clone(),
    })
  }

  pub fn from_model_hub_sync(hub: &ModelHub, spec: &ModelSpec) -> Result<Self> {
    let model = reranker_model_for(&spec.repo_id)?;
    let options = RerankInitOptions::new(model).with_cache_dir(hub.cache_dir().to_path_buf());
    let inner = TextRerank::try_new(options).map_err(|e| ModelError::ModelInitFailed(e.to_string()))?;
    Ok(Self {
      inner: Mutex::new(inner),
      model_name: spec.repo_id.clone(),
    })
  }

  pub fn model_name(&self) -> &str {
    &self.model_name
  }
}

fn reranker_model_for(repo_id: &str) -> Result<RerankerModel> {
  match repo_id {
    BGE_RERANKER_BASE_REPO => Ok(RerankerModel::BGERerankerBase),
    RERANKER_REPO_BGE_V2_M3 | RERANKER_REPO_BGE_V2_M3_ALT => Ok(RerankerModel::BGERerankerV2M3),
    RERANKER_REPO_JINA_V1_TURBO_EN => Ok(RerankerModel::JINARerankerV1TurboEn),
    RERANKER_REPO_JINA_V2_MULTILINGUAL => Ok(RerankerModel::JINARerankerV2BaseMultiligual),
    other => Err(ModelError::UnsupportedModel(format!("unsupported reranker model: {other}")).into()),
  }
}

#[async_trait::async_trait]
impl Reranker for FastEmbedReranker {
  async fn rerank(&self, query: &str, chunks: &[Chunk]) -> Result<Vec<ScoredChunk>> {
    let documents: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let mut inner = self.inner.lock().map_err(|_| RetrieveError::MutexPoisoned)?;
    let results = inner
      .rerank(query.to_string(), &documents, false, None)
      .map_err(|e| RetrieveError::RerankFailed(e.to_string()))?;

    let scored = results
      .into_iter()
      .filter_map(|r| {
        let chunk = chunks.get(r.index)?;
        Some(ScoredChunk {
          chunk: chunk.clone(),
          score: r.score,
        })
      })
      .collect();
    Ok(scored)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use semquery_core::{Chunk, Reranker};
  use tempfile::TempDir;

  #[tokio::test]
  #[ignore = "requires network; run with cargo test -- --ignored"]
  async fn test_reranker_rerank() {
    let tmp = TempDir::new().unwrap();
    let hub = ModelHub::new(tmp.path().to_path_buf());
    let spec = crate::ModelRegistry::default_reranker();

    let reranker = FastEmbedReranker::from_model_hub(&hub, &spec).await.unwrap();
    assert_eq!(reranker.model_name(), crate::BGE_RERANKER_BASE_REPO);

    let chunks = vec![
      Chunk {
        id: "a".into(),
        doc_id: "doc1".into(),
        text: "分布式共识算法".into(),
        byte_range: 0..7,
      },
      Chunk {
        id: "b".into(),
        doc_id: "doc2".into(),
        text: "今天天气不错".into(),
        byte_range: 0..6,
      },
    ];

    let scored = reranker.rerank("共识算法", &chunks).await.unwrap();
    assert_eq!(scored.len(), 2);
    assert_eq!(scored[0].chunk.id, "a");
    assert!(scored[0].score > scored[1].score);
  }
}
