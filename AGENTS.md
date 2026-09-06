# AGENTS.md — Navigation for Coding Agents

> This file is a navigation guide for coding agents working on this repository. Read it first.

## What is semquery

`semquery` is a **local-first, full RAG system** (retrieval + answer synthesis) written in Rust as a library-first multi-crate workspace:
- Millisecond-latency `search` for agents (BM25 + dense vectors + RRF + cross-encoder rerank), with zero LLM cost.
- `ask` for humans — natural-language answers with inline `[N]` citations, running on a local GGUF model or an OpenAI-compatible endpoint.
- All indexes live in a single SQLite file (`sqlite-vec` + `FTS5` + plain tables); the whole pipeline works offline.
- Chinese-optimized: chunking replicates LlamaIndex's `SentenceSplitter`; BM25 uses jieba word-level pre-tokenization.

See `README.md` for the project overview.

## Common commands

```bash
# Fastest compile check
cargo check --workspace

# Run all tests
cargo test --workspace

# Run tests for one crate
cargo test -p semquery-storage

# Format check and apply
cargo fmt --all -- --check
cargo fmt --all

# Clippy (project requires -D warnings)
cargo clippy --all-features -- -D warnings

# Run the full pre-commit suite (check + test + fmt + clippy) in one shot
./pre-commit-check.sh
```

**Notes:**
- The Rust toolchain is pinned to `1.95.0` stable in `rust-toolchain.toml`. Do not use nightly.
- On macOS, `llama-cpp-2` defaults to Metal. This project enables Metal via `GGML_METAL=ON` in `.cargo/config.toml` (for both `aarch64-apple-darwin` and `x86_64-apple-darwin`), plus `CMAKE_CXX_FLAGS="-Wno-elaborated-enum-base"` to silence a warning. Do not remove these settings.
- The first `cargo check` builds heavy C/C++ dependencies (`libsqlite3-sys`, `llama-cpp-sys-2`, `ort-sys`); 5–10 minutes is normal. Incremental builds are much faster.

## Architecture and crate relationships

```
        cli (semquery)        mcp (future)
         └───────┬─────────┘
                 ▼
            semquery            facade — the only crate library users touch
          ╱      │      ╲
  retrieve     index    synthesize        synthesize is optional (needs semquery-model/llm)
      │  ╲      │          │
      │   ╲     │          ▼
      │    ╲    │       model           GGUF / ONNX backends
      │     ╲   │      ╱
      ▼      ▼ ▼     ▼
           core                     types + traits, zero heavy dependencies
```

### Crate responsibilities and dependencies

| crate | responsibility | depends on |
|---|---|---|
| `semquery-core` | All core types + traits + error types; zero internal deps | none |
| `semquery-model` | Model registry, HF download cache, verification, inference backends (Embedder/Reranker/Llm) | core |
| `semquery-indexer` | File reading, chunking, incremental indexing, content-addressed dedup | core + storage + model(embed) |
| `semquery-storage` | SQLite `Storage` impl: documents / chunks / `vec_chunks` (sqlite-vec) / `fts_chunks` (FTS5) / model_versions | core |
| `semquery-retrieve` | BM25 + vector recall → RRF fusion → rerank; returns `SearchHit` + `ScoreExplain` | core + storage + model(rerank) |
| `semquery-synth` | Ask: build prompt → LLM → parse `[N]` citations → `Answer` | core + retrieve + model(llm) |
| `semquery` | CLI binary and `Engine` facade; exposes `init/add/index/search/ask/status` plus `Engine` for library users | all of the above |

### Layering rules (important — do not break)

- Upper layers may depend on lower layers; **lower layers must not depend on upper layers**.
- `semquery-core` does not depend on any other internal crate — it defines all traits that other crates implement. This is the key to the library-first promise: `cargo add semquery-core` does not pull in SQLite, llama.cpp, or other heavy stacks.
- `indexer` and `retrieve` do not depend on each other; both operate on data via the `Storage` trait.
- SQLite details are fully isolated within `semquery-storage`.
- `semquery-model` uses feature flags (`embed` / `rerank` / `llm`, all on by default) so consumers enable only the backends they need (avoiding "just want search but must compile llama.cpp").

## Key design decisions (do not overturn without intent)

1. **The `Storage` trait stays in `semquery-core`**, not in `semquery-storage`. This is dependency inversion: `indexer` and `retrieve` only need `semquery-core` + `semquery-model` and do not pull `rusqlite`/`sqlite-vec` at compile time. A future `semquery-storage-pg` would be a drop-in replacement.
2. **No `InMemoryStorage`**. Tests use `SqliteStorage::open_in_memory()` (SQLite `:memory:` mode, millisecond startup).
3. **`chunks.text` is the original text; `fts_chunks.text` is the jieba-tokenized space-joined text**. `StorageTx::add_fts_chunks(chunk_ids, tokenized_texts)` writes to the FTS table separately — `add_chunks` only writes the `chunks` table. `IndexTx` calls both methods inside one `begin_tx` / `commit` bracket.
4. **All mutations flow through `StorageTx`** — `Storage` is read-only (queries + `init` + `begin_tx`). This enforces transactional writes at the type level: `semquery-retrieve` holds `&Storage` and cannot write; `semquery-indexer` holds `&mut dyn StorageTx` and all four indexed tables (`documents` / `chunks` / `vec_chunks` / `fts_chunks`) plus `model_versions` commit atomically, so a re-embedding failure cannot leave the store half-written.
5. **`Document.id` is the SHA-256 of the file path** — renaming the file changes the id and triggers a reindex, keeping the logic simple.
6. **`Chunk.id` is the SHA-256 of `text`** — naturally enables content-addressed dedup and change detection.
7. **Embedding model upgrades trigger an explicit reindex**: the `model_versions` table records the current model spec for each role; the indexer compares the stored spec with the live one and forces a re-embedding when they differ, avoiding silent staleness.
8. **Invalid citations in the `ask` flow are filtered out**: after the LLM produces `[N]` markers, only those that actually appear in the provided context are kept.
9. **Not in v0.1**: MCP server, xlsx parsing, Python bindings, file-watcher auto-indexing, `semquery model` subcommand; citation precision is limited to "file + byte range" (not heading/page/row).

## Code style

- `rustfmt.toml`: `max_width=120`, `tab_spaces=2`, `chain_width=100`, `reorder_imports=true`, `merge_derives=false`.
- Private struct fields use 2-space indent.
- Error types use `thiserror`; do not hand-write `Display`.
- Async traits use `#[async_trait]`.
- Public APIs get short rustdoc (one-line `//!` module description + field comments only for non-obvious conventions, e.g. "SHA-256 of `text`", "RFC3339 UTC").
- **Never** use type-erasure shims like `as any` / `@ts-ignore` (TS concepts). The Rust equivalents are `unimplemented!()` / `todo!()` — only allowed in temporary stub methods, and must be removed before commit.
- **No deep-path references in code**: types like `semquery_core::EmbedError::Other` must be flattened to `EmbedError::Other` via `use` at the top of the file. Never nest more than two `::` levels inline — import the item and use the short name. This keeps lines short and makes dependencies explicit at the file top.

## Commit conventions

```
feat: add sqlite-vec integration
fix: handle missing model file in ModelHub
refactor: split Storage trait into sync methods
doc: update README usage examples
```

Before committing, make sure:
1. `cargo check --workspace` passes.
2. New tests pass.
3. `cargo clippy --all-features -- -D warnings` passes.
4. `cargo fmt --all -- --check` passes.

## Testing conventions

- Unit tests prefer stub embedders / stub rerankers / stub LLMs to avoid real model downloads.
- Tests that need a real model (e.g. loading a 4.5GB GGUF) are marked `#[ignore]` and run locally via `cargo test -- --ignored`, not in CI.
- Integration tests use `SqliteStorage::open_in_memory()` and need no external resources.

## Known external dependency behavior

- `sqlite-vec` registers itself process-globally via `sqlite3_auto_extension` — guarded by a `std::sync::Once` to avoid duplicate registration. See `ensure_vec_extension()` in `crates/semquery-storage/src/sqlite.rs`.
- Vectors are passed to sqlite-vec as packed native-endian `f32` byte streams; KNN queries use `WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance`.
- `llama-cpp-2` requires cmake to compile the `llama.cpp` C++ source; first build is slow. `GGML_METAL=ON` enables the Metal GPU backend on macOS (set in `.cargo/config.toml`).
- `fastembed` pulls in the ONNX runtime and model files; on first run it downloads models to `~/.cache/fastembed` or a similar directory.

## Common task navigation

| You want to | Look at |
|---|---|
| Add a new Storage backend (e.g. PostgreSQL) | `semquery-core`'s `Storage` trait + existing `SqliteStorage` as reference |
| Add a new embedding/rerank/LLM backend | `semquery-core`'s corresponding trait + `semquery-model`'s `fastembed`/`llama-cpp-2` implementations |
| Add a new file format (PDF/xlsx/docx) | `semquery-indexer`'s `reader.rs`; add a feature flag for each new extractor |
| Add a CLI subcommand | `semquery` crate's `src/main.rs`, using clap derive |
| Change the schema | `SqliteStorage::init()`'s `execute_batch` + related CRUD methods; consider a migration path. For v0.1 a simple breaking change is fine. |
| Add a unit test | Same module as the code under test, in `#[cfg(test)] mod tests`; see the existing 5 tests in `sqlite.rs` for reference |

## Workspace dependency management

- **Declare every dependency version once, in the root `Cargo.toml` under `[workspace.dependencies]`.**
  This applies to `dependencies`, `dev-dependencies`, and `optional dependencies`.

- **Sub-crate `Cargo.toml` files must not contain inline version numbers.**
  Always reference workspace-declared dependencies with `{ workspace = true }`.

- **If a sub-crate needs features or wants to mark a dependency as optional, only override those attributes; the version still comes from the workspace.**
  For example:

  ```toml
  [dependencies]
  fastembed = { workspace = true, optional = true }
  tokio-stream = { workspace = true }

  [dev-dependencies]
  tempfile = { workspace = true }
  ```

- **When adding a new dependency, first add it to the root `Cargo.toml`, then reference it from the sub-crates that need it.**

This rule applies to the entire workspace and must be followed for all future dependency changes.
