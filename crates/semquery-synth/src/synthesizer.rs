//! Citation-grounded answer synthesis over retrieved passages.

use std::sync::Arc;
use std::time::Instant;

use futures_core::Stream;
use semquery_core::{Answer, AskEvent, Citation, DocqError, Llm, Result, SearchEvent, Verbose};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::citation::parse_citations;
use crate::prompt::build_ask_prompt;

type AskEventItem = std::result::Result<AskEvent, DocqError>;
type AskEventSender = mpsc::Sender<AskEventItem>;

pub struct SynthesizerConfig {
  pub retriever: Arc<semquery_retrieve::Retriever>,
  pub llm: Arc<dyn Llm>,
  pub verbose: Verbose,
}

#[derive(Clone)]
pub struct Synthesizer {
  retriever: Arc<semquery_retrieve::Retriever>,
  llm: Arc<dyn Llm>,
  verbose: Verbose,
}

impl Synthesizer {
  pub fn new(config: SynthesizerConfig) -> Self {
    Self {
      retriever: config.retriever,
      llm: config.llm,
      verbose: config.verbose,
    }
  }

  /// Search for relevant chunks, build a prompt, call the LLM, parse
  /// citations, and return an `Answer` with sources linked back to the
  /// chunks that produced them.
  ///
  /// Citation markers `[1]`, `[2]` in the LLM output are matched against
  /// the markers generated from the retrieved hits. Invalid markers (e.g.
  /// `[3]` when only 2 chunks were retrieved) are silently dropped.
  pub async fn ask(&self, query: &str) -> Result<Answer> {
    let _total = self.verbose.start("ask");

    // ---- Retrieve top-5 chunks relevant to the query ----
    let hits = {
      let _step = self.verbose.start("retrieve");
      self.retriever.search(query, 5).await?
    };
    if hits.is_empty() {
      return Ok(Answer {
        text: String::new(),
        citations: Vec::new(),
      });
    }

    // ---- Build the prompt with numbered context blocks ----
    // Each hit becomes `[N] doc_id (bytes start-end): text` in the prompt.
    // The marker set `[1]..[N]` is the valid citation range for the LLM.
    let (valid_markers, prompt) = {
      let _step = self.verbose.start("build prompt");
      let valid_markers: Vec<String> = hits.iter().enumerate().map(|(i, _)| format!("[{}]", i + 1)).collect();
      let prompt = build_ask_prompt(query, &hits);
      (valid_markers, prompt)
    };

    // ---- Generate the answer via the LLM ----
    let raw = {
      let _step = self.verbose.start("LLM complete");
      self.llm.complete(&prompt).await?
    };

    // ---- Parse and validate citation markers from the answer ----
    let valid = {
      let _step = self.verbose.start("parse citations");
      parse_citations(&raw, &valid_markers)
    };

    // ---- Back-fill citation sources from the retrieved chunks ----
    // Each valid marker `[N]` maps to hits[N-1].chunk — we extract
    // doc_id and byte_range to produce a human-readable source string.
    let citations: Vec<Citation> = valid
      .into_iter()
      .filter_map(|marker| {
        let num: usize = marker.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
        let hit = hits.get(num.checked_sub(1)?)?;
        let chunk = &hit.chunk;
        Some(Citation {
          marker,
          source: format!(
            "{} (bytes {}-{})",
            hit.file_path.display(),
            chunk.byte_range.start,
            chunk.byte_range.end
          ),
        })
      })
      .collect();

    Ok(Answer { text: raw, citations })
  }

  /// Streaming variant of [`Synthesizer::ask`].
  ///
  /// Runs the pipeline in an independent task: emits retrieval stage events,
  /// then streams each LLM token as it is decoded, and terminates with an
  /// `AnswerComplete` event carrying the full answer plus timings. Dropping
  /// the stream cancels the pipeline.
  pub fn ask_stream(&self, query: impl Into<String>) -> impl Stream<Item = AskEventItem> + Send + 'static {
    let (tx, rx) = mpsc::channel::<AskEventItem>(32);
    let this = <Self as Clone>::clone(self);
    let query = query.into();

    tokio::spawn(async move {
      if let Err(e) = this.ask_internal(&query, &tx).await {
        let _ = tx.send(Err(e)).await;
      }
    });

    ReceiverStream::new(rx)
  }

  async fn ask_internal(&self, query: &str, tx: &AskEventSender) -> Result<()> {
    use tokio_stream::StreamExt;

    let total_start = Instant::now();
    let mut stats = semquery_core::AskStats::default();

    let _total = self.verbose.start("ask");

    // ---- Retrieve the grounding chunks, forwarding fine-grained stage events ----
    let mut search = std::pin::pin!(Arc::clone(&self.retriever).search_stream(query, 5));
    let mut hits: Vec<semquery_core::SearchHit> = Vec::new();
    while let Some(event) = search.as_mut().next().await {
      match event? {
        SearchEvent::StageStarted { stage } => {
          if tx.send(Ok(AskEvent::StageStarted { stage })).await.is_err() {
            return Ok(());
          }
        }
        SearchEvent::StageFinished { stage, elapsed_ms } => {
          if tx.send(Ok(AskEvent::StageFinished { stage, elapsed_ms })).await.is_err() {
            return Ok(());
          }
        }
        SearchEvent::Completed {
          hits: grounded,
          stats: search_stats,
        } => {
          stats.retrieve_ms = search_stats.total_ms;
          hits = grounded;
        }
      }
    }
    if tx.send(Ok(AskEvent::Hits { hits: hits.clone() })).await.is_err() {
      return Ok(());
    }
    if hits.is_empty() {
      stats.total_ms = total_start.elapsed().as_millis() as u64;
      let _ = tx
        .send(Ok(AskEvent::AnswerComplete {
          answer: Answer {
            text: String::new(),
            citations: Vec::new(),
          },
          stats,
        }))
        .await;
      return Ok(());
    }

    // ---- Build the prompt ----
    let prompt_start = Instant::now();
    let valid_markers: Vec<String> = hits.iter().enumerate().map(|(i, _)| format!("[{}]", i + 1)).collect();
    let prompt = build_ask_prompt(query, &hits);
    stats.prompt_ms = prompt_start.elapsed().as_millis() as u64;

    // ---- Generate the answer, streaming each token ----
    // The sync token callback uses `try_send`: sending a progress token into a
    // full channel drops that event rather than blocking the decode loop —
    // the full text still arrives with the terminal `AnswerComplete`.
    let llm_start = Instant::now();
    let prompt = prompt.to_string();
    let mut on_token = |piece: String| {
      let _ = tx.try_send(Ok(AskEvent::Token { delta: piece }));
    };
    let raw = self.llm.complete_stream(&prompt, &mut on_token).await?;
    stats.llm_ms = llm_start.elapsed().as_millis() as u64;

    // ---- Parse and validate citations, back-fill sources ----
    let valid = parse_citations(&raw, &valid_markers);
    let citations: Vec<Citation> = valid
      .into_iter()
      .filter_map(|marker| {
        let num: usize = marker.trim_start_matches('[').trim_end_matches(']').parse().ok()?;
        let hit = hits.get(num.checked_sub(1)?)?;
        let chunk = &hit.chunk;
        Some(Citation {
          marker,
          source: format!(
            "{} (bytes {}-{})",
            hit.file_path.display(),
            chunk.byte_range.start,
            chunk.byte_range.end
          ),
        })
      })
      .collect();

    stats.total_ms = total_start.elapsed().as_millis() as u64;
    let _ = tx
      .send(Ok(AskEvent::AnswerComplete {
        answer: Answer { text: raw, citations },
        stats,
      }))
      .await;
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use semquery_core::{Chunker, Embedder, ModelRole, ModelSpec, Storage};
  use semquery_indexer::{Indexer, IndexerConfig, JiebaSegmenter, ReaderRegistry, TextFileReader};
  use semquery_retrieve::{Retriever, RetrieverConfig};
  use semquery_storage::SqliteStorage;
  use tempfile::TempDir;

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
    fn chunk(&self, text: &str) -> Vec<semquery_core::ChunkCandidate> {
      if text.trim().is_empty() {
        return Vec::new();
      }
      vec![semquery_core::ChunkCandidate {
        text: text.to_string(),
        byte_range: 0..text.len(),
      }]
    }
  }

  struct StubLlm {
    response: String,
  }

  #[async_trait::async_trait]
  impl Llm for StubLlm {
    async fn complete(&self, _prompt: &str) -> Result<String> {
      Ok(self.response.clone())
    }
  }

  fn test_readers() -> ReaderRegistry {
    let mut reg = ReaderRegistry::new();
    reg.register(Arc::new(TextFileReader::new()));
    reg
  }

  fn test_storage() -> SqliteStorage {
    let s = SqliteStorage::open_in_memory().unwrap();
    s.init(512).unwrap();
    s
  }

  async fn seed_index(storage: &Arc<SqliteStorage>, texts: &[(&str, &str)]) {
    let tmp = TempDir::new().unwrap();
    for (filename, content) in texts.iter() {
      let path = tmp.path().join(filename);
      std::fs::write(&path, content).unwrap();
      let indexer = Indexer::new(IndexerConfig {
        chunker: Arc::new(StubChunker),
        embedder: Arc::new(StubEmbedder { dim: 512 }),
        segmenter: Arc::new(JiebaSegmenter),
        storage: storage.clone(),
        readers: test_readers(),
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
      });
      indexer.index_file(&path).await.unwrap();
    }
  }

  fn test_retriever(storage: &Arc<SqliteStorage>) -> Retriever {
    Retriever::new(RetrieverConfig {
      storage: storage.clone(),
      embedder: Arc::new(StubEmbedder { dim: 512 }),
      segmenter: Arc::new(JiebaSegmenter),
      reranker: None,
      bm25_top_k: 100,
      vector_top_k: 100,
      rrf_k: 60,
      rerank_top_n: 20,
      verbose: Verbose(false),
    })
  }

  #[tokio::test]
  async fn test_ask_with_citations() {
    let storage = Arc::new(test_storage());
    seed_index(
      &storage,
      &[("a.txt", "定价方案选坐席制"), ("b.txt", "访谈发现团队按人头预算")],
    )
    .await;

    let retriever = test_retriever(&storage);
    let llm = StubLlm {
      response: "选坐席制是因为 [1]，访谈发现 [2]。".into(),
    };

    let synth = Synthesizer::new(SynthesizerConfig {
      retriever: Arc::new(retriever),
      llm: Arc::new(llm),
      verbose: Verbose(false),
    });

    let answer = synth.ask("定价方案").await.unwrap();
    assert!(!answer.text.is_empty());
    assert_eq!(answer.citations.len(), 2);
    assert_eq!(answer.citations[0].marker, "[1]");
    assert_eq!(answer.citations[1].marker, "[2]");
  }

  #[tokio::test]
  async fn test_ask_filters_invalid_citations() {
    let storage = Arc::new(test_storage());
    seed_index(&storage, &[("a.txt", "content about pricing")]).await;

    let retriever = test_retriever(&storage);
    let llm = StubLlm {
      response: "Answer [1] and [3].".into(),
    };

    let synth = Synthesizer::new(SynthesizerConfig {
      retriever: Arc::new(retriever),
      llm: Arc::new(llm),
      verbose: Verbose(false),
    });

    let answer = synth.ask("pricing").await.unwrap();
    assert_eq!(answer.citations.len(), 1);
    assert_eq!(answer.citations[0].marker, "[1]");
  }

  #[tokio::test]
  async fn test_ask_no_hits() {
    let storage = Arc::new(test_storage());
    let retriever = test_retriever(&storage);
    let llm = StubLlm {
      response: "should not be called".into(),
    };

    let synth = Synthesizer::new(SynthesizerConfig {
      retriever: Arc::new(retriever),
      llm: Arc::new(llm),
      verbose: Verbose(false),
    });

    let answer = synth.ask("nothing matches here").await.unwrap();
    assert!(answer.text.is_empty());
    assert!(answer.citations.is_empty());
  }

  #[tokio::test]
  async fn test_ask_stream_events() {
    let storage = Arc::new(test_storage());
    seed_index(
      &storage,
      &[("a.txt", "定价方案选坐席制"), ("b.txt", "访谈发现团队按人头预算")],
    )
    .await;

    let retriever = test_retriever(&storage);
    let llm = StubLlm {
      response: "选坐席制是因为 [1]，访谈发现 [2]。".into(),
    };

    let synth = Synthesizer::new(SynthesizerConfig {
      retriever: Arc::new(retriever),
      llm: Arc::new(llm),
      verbose: Verbose(false),
    });

    let mut events: Vec<AskEvent> = Vec::new();
    {
      use tokio_stream::StreamExt;
      let mut stream = std::pin::pin!(synth.ask_stream("定价方案"));
      while let Some(event) = stream.as_mut().next().await {
        events.push(event.expect("stream must not error"));
      }
    }

    // Terminal event must be AnswerComplete with the full answer.
    let Some(AskEvent::AnswerComplete { answer, .. }) = events.last() else {
      panic!("last event must be AnswerComplete");
    };
    assert!(!answer.text.is_empty());
    assert_eq!(answer.citations.len(), 2);

    // A Token event must carry the streamed LLM output.
    let tokens: Vec<&AskEvent> = events.iter().filter(|e| matches!(e, AskEvent::Token { .. })).collect();
    assert!(!tokens.is_empty(), "at least one Token event expected");

    // A Hits event must carry the grounding context before tokens arrive.
    assert!(events.iter().any(|e| matches!(e, AskEvent::Hits { hits } if !hits.is_empty())));
  }
}
