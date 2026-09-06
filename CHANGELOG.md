## [0.2.0] - 2026-08-23

### 🚀 Features

- *(indexer)* Route single-file indexing through ReaderRegistry and add a per-extension read_file API
- *(storage)* Create vector table with dynamic dimension from embedder instead of hardcoded 512
- *(logging)* Introduce flexi_logger for file-based logging with configurable path, size rotation, and optional stderr duplication via config.toml and CLI flags
- *(verbose)* Route Verbose progress messages to the log file under the 'semquery' target and ensure they remain visible on the terminal when log duplication is disabled
- Enable Metal GPU acceleration and suppress ggml_metal_device_init log noise via early ggml_log_set no-op callback

### 🐛 Bug Fixes

- Preserve chunk-to-doc mapping via chunk_documents junction table to prevent cross-file dedup attribution errors
- Make tokenizer filename configurable in ModelEntry instead of hardcoding BGE_SMALL_ZH_V1_5_TOKENIZER_FILE
- Use atomic single-statement write for model version to avoid SQLite nested-transaction error during parallel model loading
- Enable SQLite foreign keys and WAL mode, and use little-endian for vector bytes to ensure cross-arch portability
- Skip vec_chunks creation on init(0) and pass 0 from init/add/status commands to avoid dimension mismatch when embedding model differs from default
- Increase default LLM n_ctx from 4096 to 8192 to prevent prompt truncation and preserve full retrieval context

### 🚜 Refactor

- *(indexer)* Flatten Indexer to own its component fields, and document known correctness defects
- Separate stable doc_id from file_path via document_paths and resolve file paths in SearchHit
- *(engine)* Extract base model-loading helper including ModelHub creation for open_for_index, open_for_search, and open_for_ask
- *(engine)* Extract base model-loading helper including ModelHub creation for open_for_index, open_for_search, and open_for_ask
- *(retriever)* Flatten Retriever to own its component fields directly from RetrieverConfig
- *(semquery-retrieve)* Split hybrid search pipeline into focused helpers; feat(fusion): generalize RRF to multiple recall channels; doc(semquery-retrieve): document each retrieval stage and score semantics
- *(semquery-synth)* Move Synthesizer and SynthesizerConfig into synthesizer.rs and flatten config fields directly onto the struct
- Replace all Other(String) error variants with structured variants across 6 error enums
- Extract Indexer struct, IndexStats, and tests into indexer.rs, and split run_command into per-subcommand handler functions
- Replace ModelSpec.role String with ModelRole enum across all crates

### 📚 Documentation

- Update docs

### ⚡ Performance

- Parallelize BM25 and vector recall in Retriever::search via spawn_blocking and tokio::join!
- Parallelize reranker and LLM loading in Engine::open_for_ask via spawn_blocking and tokio::join!
- Cache token_count in chunker, batch-petch document hashes, and pass is_update to eliminate redundant DB queries in indexer
- Switch default reranker from BGE-reranker-base to jina-reranker-v1-turbo-en for 4x faster reranking
- Switch default reranker from BGE-reranker-base to jina-reranker-v1-turbo-en for 4x faster reranking
## [0.1.0] - 2026-08-15

### 🚀 Features

- Scaffold workspace with semquery-core types, traits, and error taxonomy
- Implement SQLite-backed Storage with documents, chunks, and model versions CRUD
- Integrate sqlite-vec and FTS5 vector and full-text search into SqliteStorage
- Add ModelRegistry defaults and ModelHub with HuggingFace download/cache and model version recording
- Add FastEmbedEmbedder with ModelSpec-driven model selection and TextReader for recursive file scanning
- Add SentenceSplitter, jieba WordSegmenter, and Indexer with incremental reindex and cascade delete
- Add Retriever with BM25 and vector recall fused via RRF with jieba-segmented query and unified score direction
- Add cross-encoder reranker with optional rerank integration in Retriever and centralize model repo constants
- Add GgufLlm with LlmConfig for local GGUF inference with configurable sampling parameters
- Add Synthesizer with prompt builder, citation parser, and LLM answer generation grounded in retrieved chunks
- Add Engine facade with DI components, collection persistence, model version recording, and model-constrained chunk sizing
- *(cli)* Implement lazy-loading Engine facade and interactive CLI
- *(config)* Add workspace TOML config and platform-aware default directories
- *(cli)* Ensure the global default config.toml is created and loaded from the system config directory on every command, independent of the workspace directory
- *(cli)* Support custom config files via --config/-c and auto-create the global default config.toml on every command
- *(cli)* Support custom config files via --config/-c and auto-create the global default config.toml on every command
- *(cli)* Add -v/--verbose global flag and per-step timing output for index, search and ask
- *(indexer)* Abstract FileReader trait and ReaderRegistry dispatcher for pluggable PDF/DOC/text readers; refactor(core): move DocumentSource to semquery-core for cross-crate reuse
- *(indexer)* Add PDF support via pdf-extract behind the `pdf` feature and register PdfReader in default readers; doc(testdata): reorganize bundled example docs into testdata/md and add a sample PDF
- *(indexer)* Add DOCX reader with zip + quick-xml

### 🐛 Bug Fixes

- *(retrieve)* Quote FTS5 query tokens to avoid syntax errors on hyphens and special characters, add verbose index progress and timing, and document the bundled testdata/en example
- *(model)* Raise BGE-small-zh max tokens from 512 to 1024; fix(model): align llama.cpp n_batch with n_ctx and truncate over-budget prompts to prevent GGML_ASSERT aborts on large chunks
- *(build)* Vendor patched esaxx-rs to use dynamic CRT on Windows and avoid LNK2038 linker mismatch

### 🚜 Refactor

- Split Storage into read-only Storage and write-only StorageTx with all mutations flowing through transactions
- Flatten deep-path type references to top-level imports and document the convention in AGENTS.md
- *(model)* Rename default model registry constants to reflect model and file names, and prepare workspace crates for crates.io publishing
- *(indexer)* Batch embedding calls up to 500 chunks and extract prepare_file helper to reduce ONNX invocation overhead; doc(readme): polish bundled testdata attribution and grammar
- *(indexer)* Reorganize reader module by moving TextFileReader to reader/txt_reader.rs and ReaderRegistry to reader_registry.rs for clearer separation of format readers and dispatch logic

### 📚 Documentation

- Update docs
- Update docs
- Update docs
- Update docs
- Update docs
- Update docs
- Update docs
- Update docs
- Update docs

### ⚙️ Miscellaneous Tasks

- *(release)* Prepare all workspace crates for crates.io with required metadata, versioned internal dependencies, a version-aware publish script, and fix clippy warnings
- *(ci)* Add release workflow to build multi-platform binaries and generate changelog
- *(release)* Build x86_64-apple-darwin on macos-latest via cross-compilation
- *(release)* Ensure cross-compilation target is installed for rust-toolchain.toml
- *(release)* Drop x86_64-apple-darwin prebuilt binary due to missing ONNX Runtime binaries
