use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use semquery_core::{ModelError, ModelSpec, Result, Storage};

#[derive(Clone)]
pub struct ModelHub {
  cache_dir: PathBuf,
}

impl ModelHub {
  pub fn new(cache_dir: PathBuf) -> Self {
    Self { cache_dir }
  }

  pub fn cache_dir(&self) -> &Path {
    &self.cache_dir
  }

  pub async fn ensure(&self, spec: &ModelSpec, storage: &dyn Storage) -> Result<PathBuf> {
    let path = self.resolve(spec).await?;
    storage.set_model_version_atomic(spec.role, spec)?;
    Ok(path)
  }

  pub async fn resolve(&self, spec: &ModelSpec) -> Result<PathBuf> {
    let api = ApiBuilder::new()
      .with_cache_dir(self.cache_dir.clone())
      .with_progress(true)
      .build()
      .map_err(|e| ModelError::HubApiFailed(e.to_string()))?;

    let repo = Repo::with_revision(spec.repo_id.clone(), RepoType::Model, spec.revision.clone());
    api
      .repo(repo)
      .get(&spec.filename)
      .map_err(|e| ModelError::DownloadFailed(e.to_string()))
      .map_err(Into::into)
  }

  pub fn ensure_sync(&self, spec: &ModelSpec, storage: &dyn Storage) -> Result<PathBuf> {
    let path = self.resolve_sync(spec)?;
    storage.set_model_version_atomic(spec.role, spec)?;
    Ok(path)
  }

  pub fn resolve_sync(&self, spec: &ModelSpec) -> Result<PathBuf> {
    let api = ApiBuilder::new()
      .with_cache_dir(self.cache_dir.clone())
      .with_progress(true)
      .build()
      .map_err(|e| ModelError::HubApiFailed(e.to_string()))?;

    let repo = Repo::with_revision(spec.repo_id.clone(), RepoType::Model, spec.revision.clone());
    api
      .repo(repo)
      .get(&spec.filename)
      .map_err(|e| ModelError::DownloadFailed(e.to_string()))
      .map_err(Into::into)
  }
}
