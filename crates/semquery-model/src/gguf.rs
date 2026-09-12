use std::num::NonZeroU32;

use encoding_rs::UTF_8;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use semquery_core::{Llm, LlmConfig, LlmError, ModelSpec, Result};

use crate::ModelHub;

unsafe extern "C" {
  fn ggml_log_set(
    callback: Option<
      unsafe extern "C" fn(level: i32, text: *const std::os::raw::c_char, user_data: *mut std::os::raw::c_void),
    >,
    user_data: *mut std::os::raw::c_void,
  );
}

unsafe extern "C" fn void_log(_level: i32, _text: *const std::os::raw::c_char, _user_data: *mut std::os::raw::c_void) {}

fn silence_ggml_logs() {
  unsafe {
    ggml_log_set(Some(void_log), std::ptr::null_mut());
  }
}

pub struct GgufLlm {
  backend: LlamaBackend,
  model: LlamaModel,
  chat_template: LlamaChatTemplate,
  config: LlmConfig,
}

impl GgufLlm {
  pub async fn from_model_hub(hub: &ModelHub, spec: &ModelSpec, config: &LlmConfig) -> Result<Self> {
    let path = hub.resolve(spec).await?;

    silence_ggml_logs();
    let mut backend = LlamaBackend::init().map_err(|e| LlmError::BackendInit(e.to_string()))?;
    backend.void_logs();
    let model_params = LlamaModelParams::default();
    let model =
      LlamaModel::load_from_file(&backend, &path, &model_params).map_err(|e| LlmError::ModelLoad(e.to_string()))?;
    let chat_template = model.chat_template(None).map_err(|e| LlmError::InferenceFailed(e.to_string()))?;

    Ok(Self {
      backend,
      model,
      chat_template,
      config: config.clone(),
    })
  }

  pub fn from_model_hub_sync(hub: &ModelHub, spec: &ModelSpec, config: &LlmConfig) -> Result<Self> {
    let path = hub.resolve_sync(spec)?;

    silence_ggml_logs();
    let mut backend = LlamaBackend::init().map_err(|e| LlmError::BackendInit(e.to_string()))?;
    backend.void_logs();
    let model_params = LlamaModelParams::default();
    let model =
      LlamaModel::load_from_file(&backend, &path, &model_params).map_err(|e| LlmError::ModelLoad(e.to_string()))?;
    let chat_template = model.chat_template(None).map_err(|e| LlmError::InferenceFailed(e.to_string()))?;

    Ok(Self {
      backend,
      model,
      chat_template,
      config: config.clone(),
    })
  }
}

#[async_trait::async_trait]
impl Llm for GgufLlm {
  async fn complete(&self, prompt: &str) -> Result<String> {
    let messages = [
      LlamaChatMessage::new("system".into(), self.config.system_prompt.clone())
        .map_err(|e| LlmError::InferenceFailed(e.to_string()))?,
      LlamaChatMessage::new("user".into(), prompt.to_string()).map_err(|e| LlmError::InferenceFailed(e.to_string()))?,
    ];

    let formatted = self
      .model
      .apply_chat_template(&self.chat_template, &messages, true)
      .map_err(|e| LlmError::InferenceFailed(e.to_string()))?;

    let tokens = self
      .model
      .str_to_token(&formatted, AddBos::Always)
      .map_err(|e| LlmError::InferenceFailed(e.to_string()))?;

    let ctx_params = LlamaContextParams::default()
      .with_n_ctx(NonZeroU32::new(self.config.n_ctx))
      .with_n_batch(self.config.n_ctx);
    let mut ctx = self
      .model
      .new_context(&self.backend, ctx_params)
      .map_err(|e| LlmError::InferenceFailed(e.to_string()))?;

    // Reserve room for the generated answer so we do not overflow the KV cache.
    let max_prompt_tokens = (self.config.n_ctx as usize).saturating_sub(self.config.max_tokens).max(1);
    let prompt_tokens = if tokens.len() > max_prompt_tokens {
      eprintln!(
        "[semquery] warning: prompt is {} tokens but context budget is {}; truncating from the beginning",
        tokens.len(),
        max_prompt_tokens
      );
      &tokens[tokens.len() - max_prompt_tokens..]
    } else {
      &tokens[..]
    };

    let mut batch = LlamaBatch::new(prompt_tokens.len() + self.config.max_tokens, 1);
    for (i, token) in prompt_tokens.iter().enumerate() {
      batch
        .add(*token, i as i32, &[0], i == prompt_tokens.len() - 1)
        .map_err(|e| LlmError::InferenceFailed(e.to_string()))?;
    }

    ctx.decode(&mut batch).map_err(|e| LlmError::InferenceFailed(e.to_string()))?;

    let mut sampler = LlamaSampler::chain_simple([
      LlamaSampler::temp(self.config.temperature),
      LlamaSampler::top_p(self.config.top_p, 1),
      LlamaSampler::dist(self.config.seed),
    ]);

    let mut output = String::new();
    let mut decoder = UTF_8.new_decoder();

    for step in 0..self.config.max_tokens {
      let pos = prompt_tokens.len() as i32 + step as i32;
      let token = sampler.sample(&ctx, batch.n_tokens() - 1);

      if self.model.is_eog_token(token) {
        break;
      }

      let piece = self
        .model
        .token_to_piece(token, &mut decoder, true, None)
        .map_err(|e| LlmError::InferenceFailed(e.to_string()))?;
      output.push_str(&piece);

      batch.clear();
      batch.add(token, pos, &[0], true).map_err(|e| LlmError::InferenceFailed(e.to_string()))?;

      ctx.decode(&mut batch).map_err(|e| LlmError::InferenceFailed(e.to_string()))?;
    }

    Ok(output)
  }

  async fn complete_stream(&self, prompt: &str, on_token: &mut (dyn FnMut(String) + Send + Sync)) -> Result<String> {
    let messages = [
      LlamaChatMessage::new("system".into(), self.config.system_prompt.clone())
        .map_err(|e| LlmError::InferenceFailed(e.to_string()))?,
      LlamaChatMessage::new("user".into(), prompt.to_string()).map_err(|e| LlmError::InferenceFailed(e.to_string()))?,
    ];

    let formatted = self
      .model
      .apply_chat_template(&self.chat_template, &messages, true)
      .map_err(|e| LlmError::InferenceFailed(e.to_string()))?;

    let tokens = self
      .model
      .str_to_token(&formatted, AddBos::Always)
      .map_err(|e| LlmError::InferenceFailed(e.to_string()))?;

    let ctx_params = LlamaContextParams::default()
      .with_n_ctx(NonZeroU32::new(self.config.n_ctx))
      .with_n_batch(self.config.n_ctx);
    let mut ctx = self
      .model
      .new_context(&self.backend, ctx_params)
      .map_err(|e| LlmError::InferenceFailed(e.to_string()))?;

    // Reserve room for the generated answer so we do not overflow the KV cache.
    let max_prompt_tokens = (self.config.n_ctx as usize).saturating_sub(self.config.max_tokens).max(1);
    let prompt_tokens = if tokens.len() > max_prompt_tokens {
      eprintln!(
        "[semquery] warning: prompt is {} tokens but context budget is {}; truncating from the beginning",
        tokens.len(),
        max_prompt_tokens
      );
      &tokens[tokens.len() - max_prompt_tokens..]
    } else {
      &tokens[..]
    };

    let mut batch = LlamaBatch::new(prompt_tokens.len() + self.config.max_tokens, 1);
    for (i, token) in prompt_tokens.iter().enumerate() {
      batch
        .add(*token, i as i32, &[0], i == prompt_tokens.len() - 1)
        .map_err(|e| LlmError::InferenceFailed(e.to_string()))?;
    }

    ctx.decode(&mut batch).map_err(|e| LlmError::InferenceFailed(e.to_string()))?;

    let mut sampler = LlamaSampler::chain_simple([
      LlamaSampler::temp(self.config.temperature),
      LlamaSampler::top_p(self.config.top_p, 1),
      LlamaSampler::dist(self.config.seed),
    ]);

    let mut output = String::new();
    let mut decoder = UTF_8.new_decoder();

    for step in 0..self.config.max_tokens {
      let pos = prompt_tokens.len() as i32 + step as i32;
      let token = sampler.sample(&ctx, batch.n_tokens() - 1);

      if self.model.is_eog_token(token) {
        break;
      }

      let piece = self
        .model
        .token_to_piece(token, &mut decoder, true, None)
        .map_err(|e| LlmError::InferenceFailed(e.to_string()))?;
      output.push_str(&piece);
      on_token(piece.to_string());

      batch.clear();
      batch.add(token, pos, &[0], true).map_err(|e| LlmError::InferenceFailed(e.to_string()))?;

      ctx.decode(&mut batch).map_err(|e| LlmError::InferenceFailed(e.to_string()))?;
    }

    Ok(output)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::ModelRegistry;
  use semquery_core::Llm;
  use tempfile::TempDir;

  #[tokio::test]
  #[ignore = "requires network + ~4.5GB model download; run with cargo test -- --ignored"]
  async fn test_llm_complete() {
    let tmp = TempDir::new().unwrap();
    let hub = ModelHub::new(tmp.path().to_path_buf());
    let spec = ModelRegistry::default_llm();

    let llm = GgufLlm::from_model_hub(&hub, &spec, &LlmConfig::default()).await.unwrap();
    let output = llm.complete("What is 2+3? Answer briefly.").await.unwrap();
    assert!(!output.is_empty());
    assert!(output.contains("5") || output.contains("五"));
  }
}
