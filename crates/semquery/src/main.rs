mod config;
mod engine;

use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use config::{DocqConfig, LoggingConfig};

use clap::{Parser, Subcommand};
use flexi_logger::{Cleanup, Criterion, DeferredNow, Duplicate, FileSpec, Logger, Naming, Record, WriteMode};
use semquery_core::{EngineStatus, Storage, Verbose};
use semquery_storage::SqliteStorage;
use serde::Serialize;

pub use engine::{Engine, EngineComponents, EngineConfig};

#[derive(Parser)]
#[command(
  name = "semq",
  version,
  about = "Local-first RAG: hybrid search and cited answers over your documents (downloads models on first use)"
)]
struct Cli {
  /// Workspace directory path (default: ~/.config/semq on Unix, %LOCALAPPDATA%\semq on Windows).
  #[arg(long, global = true)]
  workspace: Option<PathBuf>,

  /// Model cache directory (default: ~/.cache/semq/models).
  #[arg(long, global = true)]
  model_cache: Option<PathBuf>,

  /// Path to a custom configuration file (default: ~/.config/semq/config.toml).
  #[arg(short = 'c', long, global = true)]
  config: Option<PathBuf>,

  /// Enable verbose progress output (use -vv for even more detail).
  #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count)]
  verbose: u8,

  /// Log file path (default: <workspace>/semq.log). Overrides the config file.
  #[arg(long, global = true)]
  log_file: Option<PathBuf>,

  /// Also print log messages to stderr while writing to the log file.
  #[arg(long, global = true)]
  log_stdout: bool,

  #[command(subcommand)]
  command: Commands,
}

#[derive(Subcommand)]
enum Commands {
  /// Initialize a new workspace.
  Init,
  /// Add a collection (a directory to be indexed).
  Add {
    /// Directory path to index.
    path: PathBuf,
    /// Name for this collection.
    #[arg(long)]
    name: String,
  },
  /// Build or update the index.
  Index {
    /// Only index the specified collection.
    #[arg(long)]
    collection: Option<String>,
  },
  /// Search for passages (zero LLM cost).
  Search {
    /// Query string.
    query: String,
    /// Number of results to return.
    #[arg(long, default_value_t = 5)]
    top_k: usize,
    /// Show score breakdown.
    #[arg(long)]
    explain: bool,
    /// Output JSON to stdout.
    #[arg(long)]
    json: bool,
  },
  /// Ask a question and get a cited answer.
  Ask {
    /// Question string.
    query: String,
    /// Output JSON to stdout.
    #[arg(long)]
    json: bool,
  },
  /// Show workspace status.
  Status {
    /// Output JSON to stdout.
    #[arg(long)]
    json: bool,
  },
}

fn default_workspace() -> PathBuf {
  // Use XDG-style config directories on Unix and the standard local app data
  // directory on Windows. This keeps the workspace (config.toml + semq.db) in
  // the conventional per-platform config location.
  #[cfg(target_os = "macos")]
  {
    dirs::home_dir().unwrap_or_default().join(".config").join("semquery")
  }
  #[cfg(target_os = "linux")]
  {
    dirs::config_dir().unwrap_or_default().join("semquery")
  }
  #[cfg(target_os = "windows")]
  {
    dirs::config_local_dir().unwrap_or_default().join("semquery")
  }
  #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
  {
    dirs::home_dir().unwrap_or_default().join(".semq")
  }
}

/// Default directory for the global `config.toml`.
/// This is independent from the workspace (data) directory: configuration is
/// always loaded from here, even when `--workspace` points somewhere else.
fn default_config_dir() -> PathBuf {
  // Keep the global config in the conventional per-platform config location.
  #[cfg(target_os = "macos")]
  {
    dirs::home_dir().unwrap_or_default().join(".config").join("semquery")
  }
  #[cfg(target_os = "linux")]
  {
    dirs::config_dir().unwrap_or_default().join("semquery")
  }
  #[cfg(target_os = "windows")]
  {
    dirs::config_local_dir().unwrap_or_default().join("semquery")
  }
  #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
  {
    dirs::home_dir().unwrap_or_default().join(".semq")
  }
}

fn default_model_cache() -> PathBuf {
  dirs::home_dir().unwrap_or_default().join(".cache").join("semquery").join("models")
}

#[derive(Serialize)]
struct ErrorResponse {
  error: String,
}

fn print_json<T: Serialize>(value: &T) {
  println!(
    "{}",
    serde_json::to_string_pretty(value).unwrap_or_else(|e| format!("{{\"error\": \"json: {e}\"}}"))
  );
}

fn print_error_json(msg: &str) {
  eprintln!(
    "{}",
    serde_json::to_string(&ErrorResponse { error: msg.to_string() }).unwrap_or_default()
  );
}

#[tokio::main]
async fn main() {
  let cli = Cli::parse();

  let workspace = cli.workspace.unwrap_or_else(default_workspace);
  let model_cache = cli.model_cache.unwrap_or_else(default_model_cache);
  let config_result = match cli.config {
    Some(ref path) => DocqConfig::load_from_file(path),
    None => ensure_config(),
  };
  let config = match config_result {
    Ok(cfg) => cfg,
    Err(e) => {
      print_error_json(&e.to_string());
      process::exit(1);
    }
  };

  let _logger = match init_logger(&config.logging, &workspace, cli.log_file.as_deref(), cli.log_stdout) {
    Ok(handle) => handle,
    Err(e) => {
      print_error_json(&e.to_string());
      process::exit(1);
    }
  };

  log::info!("semquery v{} starting", env!("CARGO_PKG_VERSION"));

  let logs_to_terminal = cli.log_stdout || config.logging.duplicate_to_stderr;
  semquery_core::set_log_terminal_output(logs_to_terminal);

  let verbose = Verbose(cli.verbose > 0);

  if let Err(e) = run_command(&cli.command, &workspace, &model_cache, config, verbose).await {
    print_error_json(&e.to_string());
    process::exit(1);
  }
}

/// Initialize the file logger.
///
/// The log file path is resolved in this order:
/// 1. `--log-file` CLI argument.
/// 2. `logging.file` from `config.toml`.
/// 3. `<workspace>/semq.log` as the default.
///
/// Relative paths are resolved against the workspace. The log file is rotated
/// by size and old rotated files are cleaned up automatically.
/// Custom log format that prints the target (rather than the module path)
/// so that verbose progress messages can be emitted with a consistent
/// `LEVEL [semquery] message` appearance.
fn log_format(w: &mut dyn std::io::Write, _now: &mut DeferredNow, record: &Record) -> std::io::Result<()> {
  write!(w, "{} [{}] {}", record.level(), record.target(), record.args())
}

fn init_logger(
  logging: &LoggingConfig,
  workspace: &Path,
  log_file: Option<&Path>,
  log_stdout: bool,
) -> anyhow::Result<flexi_logger::LoggerHandle> {
  let log_path = log_file
    .map(PathBuf::from)
    .or_else(|| logging.file.clone())
    .unwrap_or_else(|| workspace.join("semq.log"));
  let log_path = if log_path.is_absolute() {
    log_path
  } else {
    workspace.join(log_path)
  };

  let parent = log_path.parent().unwrap_or_else(|| Path::new("."));
  fs::create_dir_all(parent)?;

  let directory = log_path.parent().and_then(|p| p.to_str()).unwrap_or(".");
  let basename = log_path.file_stem().and_then(|s| s.to_str()).unwrap_or("semquery");

  let duplicate = if log_stdout || logging.duplicate_to_stderr {
    Duplicate::All
  } else {
    Duplicate::None
  };

  let rotation_size = logging.rotation_size_mb as u64 * 1024 * 1024;

  let logger = Logger::try_with_env_or_str(&logging.level)?
    .log_to_file(FileSpec::default().directory(directory).basename(basename))
    .rotate(
      Criterion::Size(rotation_size),
      Naming::Numbers,
      Cleanup::KeepLogFiles(logging.max_files),
    )
    .duplicate_to_stderr(duplicate)
    .format(log_format)
    .write_mode(WriteMode::BufferAndFlush)
    .start()
    .map_err(|e| anyhow::anyhow!("init logger: {e}"))?;

  Ok(logger)
}

/// Ensure the global config directory and `config.toml` exist.
/// If the config file is missing, write a default one and return it.
fn ensure_config() -> anyhow::Result<DocqConfig> {
  let config_dir = default_config_dir();
  fs::create_dir_all(&config_dir)?;
  let config_path = DocqConfig::path(&config_dir);
  if config_path.exists() {
    DocqConfig::load(&config_dir)
  } else {
    let cfg = DocqConfig::default();
    fs::write(&config_path, cfg.to_toml()?)
      .map_err(|e| anyhow::anyhow!("write default config {}: {e}", config_path.display()))?;
    Ok(cfg)
  }
}

async fn run_command(
  cmd: &Commands,
  workspace: &Path,
  model_cache: &Path,
  config: DocqConfig,
  verbose: Verbose,
) -> anyhow::Result<()> {
  fs::create_dir_all(workspace)?;

  match cmd {
    Commands::Init => run_init(workspace),
    Commands::Add { path, name } => run_add(workspace, path, name),
    Commands::Status { json } => run_status(workspace, *json),
    Commands::Index { collection } => {
      run_index(workspace, model_cache, config.clone(), verbose, collection.as_deref()).await
    }
    Commands::Search {
      query,
      top_k,
      explain,
      json,
    } => {
      run_search(
        workspace,
        model_cache,
        config.clone(),
        verbose,
        query,
        *top_k,
        *explain,
        *json,
      )
      .await
    }
    Commands::Ask { query, json } => run_ask(workspace, model_cache, config.clone(), verbose, query, *json).await,
  }
}

fn run_init(workspace: &Path) -> anyhow::Result<()> {
  fs::create_dir_all(workspace)?;
  let storage = SqliteStorage::open_workspace(workspace)?;
  storage.init(0)?;
  println!("Initialized workspace at {}", workspace.display());
  Ok(())
}

fn run_add(workspace: &Path, path: &Path, name: &str) -> anyhow::Result<()> {
  let storage = open_storage(workspace)?;
  let canonical = fs::canonicalize(path)?;
  let path_str = canonical.to_string_lossy().to_string();
  let mut tx = storage.begin_tx()?;
  tx.add_collection(name, &path_str)?;
  tx.commit()?;
  println!("Added collection '{}' -> {}", name, canonical.display());
  Ok(())
}

fn run_status(workspace: &Path, json: bool) -> anyhow::Result<()> {
  let storage = open_storage(workspace)?;
  let docs = storage.list_documents()?;
  let collections = storage.list_collections()?;
  let chunks = storage.count_chunks()?;
  let status = EngineStatus {
    documents: docs.len(),
    chunks,
    collections,
  };
  if json {
    print_json(&status);
  } else {
    println!("Workspace: {}", workspace.display());
    println!("Documents: {}", status.documents);
    println!("Chunks: {}", status.chunks);
    println!("Collections: {}", status.collections.len());
    for c in &status.collections {
      println!("  {} -> {}", c.name, c.path.display());
    }
  }
  Ok(())
}

async fn run_index(
  workspace: &Path,
  model_cache: &Path,
  config: DocqConfig,
  verbose: Verbose,
  collection: Option<&str>,
) -> anyhow::Result<()> {
  let engine = Engine::open_for_index(engine_config(workspace, model_cache, config, verbose)).await?;
  let stats = if let Some(name) = collection {
    engine.index_one(name).await?
  } else {
    engine.index().await?
  };
  println!(
    "Indexed {} files ({} chunks, {} skipped, {} removed)",
    stats.files_indexed, stats.chunks_indexed, stats.files_skipped, stats.files_removed
  );
  Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_search(
  workspace: &Path,
  model_cache: &Path,
  config: DocqConfig,
  verbose: Verbose,
  query: &str,
  top_k: usize,
  explain: bool,
  json: bool,
) -> anyhow::Result<()> {
  let engine = Engine::open_for_search(engine_config(workspace, model_cache, config, verbose)).await?;
  let hits = engine.search(query, top_k).await?;

  if json {
    let output: Vec<serde_json::Value> = hits
      .iter()
      .map(|h| {
        if explain {
          serde_json::to_value(h).unwrap_or_default()
        } else {
          serde_json::json!({
            "chunk": h.chunk,
            "score": h.score,
          })
        }
      })
      .collect();
    print_json(&serde_json::json!({ "hits": output }));
  } else {
    for (i, hit) in hits.iter().enumerate() {
      println!("[{}] {:.4} {}", i + 1, hit.score, hit.chunk.text.trim());
      if explain {
        let e = &hit.explain;
        println!(
          "    bm25={:?} vec={:?} rrf={:?} rerank={:?}",
          e.bm25_score, e.vector_score, e.rrf_score, e.rerank_score
        );
      }
    }
    if hits.is_empty() {
      println!("No results found.");
    }
  }
  Ok(())
}

async fn run_ask(
  workspace: &Path,
  model_cache: &Path,
  config: DocqConfig,
  verbose: Verbose,
  query: &str,
  json: bool,
) -> anyhow::Result<()> {
  use tokio_stream::StreamExt;

  let engine = Engine::open_for_ask(engine_config(workspace, model_cache, config, verbose)).await?;
  let mut stream = std::pin::pin!(engine.ask_stream(query)?);

  let mut buffered_text = String::new();
  let mut answer: Option<semquery_core::Answer> = None;

  while let Some(event) = stream.as_mut().next().await {
    match event? {
      semquery_core::AskEvent::Token { delta } => {
        if !json {
          print!("{delta}");
          use std::io::Write;
          std::io::stdout().flush().ok();
        }
        buffered_text.push_str(&delta);
      }
      semquery_core::AskEvent::AnswerComplete { answer: a, .. } => {
        answer = Some(a);
      }
      _ => {}
    }
  }

  if json {
    match answer {
      Some(a) => print_json(&a),
      None => {
        print_json(&semquery_core::Answer {
          text: buffered_text,
          citations: Vec::new(),
        });
      }
    }
  } else {
    println!();
    if let Some(a) = answer
      && !a.citations.is_empty()
    {
      println!("\nSources:");
      for c in &a.citations {
        println!("  {} {}", c.marker, c.source);
      }
    }
  }
  Ok(())
}

fn open_storage(workspace: &Path) -> anyhow::Result<SqliteStorage> {
  let storage = SqliteStorage::open_workspace(workspace)?;
  storage.init(0)?;
  Ok(storage)
}

fn engine_config(workspace: &Path, model_cache: &Path, config: DocqConfig, verbose: Verbose) -> EngineConfig {
  EngineConfig {
    workspace_path: workspace.to_path_buf(),
    model_cache_dir: model_cache.to_path_buf(),
    config,
    verbose,
  }
}
