# semquery 性能优化方向

> 基于当前代码结构梳理的索引、查询、问答及模型推理层的优化机会。先记录方向，后续再逐步落地。

---

## 1. 索引性能（Indexer）

当前索引是**单线程、顺序、先读完整个目录再处理**的流水线，存在以下瓶颈：

| 问题 | 影响 | 优化思路 |
|---|---|---|
| `ReaderRegistry::read_dir` 把所有文件内容先收集到 `Vec<DocumentSource>` | 大目录内存峰值高，chunk/embedding 阶段被阻塞 | 改成流式：边 WalkDir 边读取、分块、向量化 ✅ |
| 每个文件都同步查一次 `storage.get_document(&doc_id)` 做去重 | N+1 查询，串行访问 Storage | 批量读一个 collection 下所有 `(doc_id, content_hash)`，一次判断是否需要跳过 |
| `flush_batch` 虽然把 embedding 批处理了 500 个 chunk，但写库时仍是**每个文件一个事务** | 抵消 batch 收益，fsync 次数多 | 每 N 个文件用一个事务写入，减少 fsync 次数 ✅ |
| `index_directory` 是 plain `for` 循环处理文件 | CPU 分词和 I/O 没重叠 | 用 `tokio::task::spawn_blocking` 或 rayon 做读取+分块，通过有界 channel 喂给 embedding 阶段 ✅ |
| `SentenceSplitter` 里 `token_count()` 对每个句子/单元反复调用 tokenizer | 分块阶段 CPU 占用高 | 只 tokenize 一次，用 token span 驱动切分，或缓存每个单元的 token count |
| `sha256_hex` 逐字节 `format!` 分配字符串 | 大文件哈希格式化有额外开销 | 用固定 hex 表 + 预分配 `String` 单次写入 ✅ |
| `index_file` 总是只 batch 一个文件 | 单文件 API 无法均摊 embedding/存储开销 | 提供公开 batched API，或让 `index_file` 内部保留跨调用缓冲 |

**快速 win**：把 batch 写入合并成单事务 + SQLite WAL 模式，这两行改动能明显提升大批量索引吞吐。

---

## 2. 查询/检索性能（Retriever）

| 问题 | 影响 | 优化思路 |
|---|---|---|
| BM25（`search_text`）和向量召回（`search_vectors`）是**顺序执行** | 延迟相加 | 并发执行后合并，再用 RRF 融合 |
| 查询向量化是 `embed(&[query])` 然后取第一个 | 多一次 Vec 分配，API 不自然 | 加 `Embedder::embed_one` 直接返回 `Vec<f32>` |
| RRF 用了多个 `HashMap` 中转，rerank 时克隆整个 `Chunk` | 堆分配和内存拷贝较多 | 用一个 `HashMap<chunk_id, ScoreAccum>` 就地累加；rerank 传引用或预分配 |
| 默认固定 `top_k = 100`，没有阈值截断 | irrelevant 结果也参与后续 rerank | 加 score threshold 或按场景减小 top_k |
| `get_chunks` 用动态 `IN (...)` 且没有数量上限 | SQLite 默认最多 999 个 host parameters，大 rerank set 可能报错 | 把 ID 列表切成 ≤900 的批次查询 |

**快速 win**：BM25 与向量召回并行 + 合并 HashMap，能直接降低 `search` 延迟。

---

## 3. 问答性能（Synthesizer / Ask）

| 问题 | 影响 | 优化思路 |
|---|---|---|
| `top_k` 硬编码为 5，没有上下文预算控制 | 上下文可能太多或太少，甚至超过 LLM 窗口 | 在 `SynthesizerConfig` 暴露 `context_top_k` 和 `max_context_tokens` |
| `build_ask_prompt` 直接把检索结果拼接，不做 token 预算检查 | 超过 `n_ctx` 时会在模型侧被截断 | 先算 prompt token 数，从**上下文末尾**截断，保留 system prompt 和 query |
| 每次 ask 都重新 embed + 检索 | 相同问题重复计算 | 可选的 query-result cache（按规范化 query + TTL） |
| LLM 侧截断是从 token 列表开头切 | 会丢掉 system prompt，模型行为变差 | 截断时保留 system prompt 和最终 query，只压缩中间上下文 |

**快速 win**：上下文 token 预算 + 从尾部截断，这是正确性和性能兼顾的改动。

---

## 4. 模型推理层（semquery-model）

| 问题 | 影响 | 优化思路 |
|---|---|---|
| `FastEmbedEmbedder` 和 `FastEmbedReranker` 都被 `Mutex` 包住 | 并发请求会串行等锁 | 确认 `TextEmbedding` 是否内部线程安全；如果是就去掉 Mutex，否则做实例池 |
| `GgufLlm::complete` 每次调用都新建 `LlamaContext` | KV cache 被丢弃，context 创建本身也重 | 按模型缓存/池化 `LlamaContext`，调用间 reset |
| 默认 CPU-only，没有 Metal/CUDA 控制 | GPU 机器性能没发挥 | 在 `LlmConfig` 暴露 `n_gpu_layers`、`n_threads`、`n_batch` |
| LLM 没有 streaming | 首 token 延迟等于完整响应延迟 | 加 `complete_stream` 返回 token 流 |
| `Engine` 每次 CLI 调用都重新构造 `ModelHub`、下载/加载模型 | 冷启动几秒 | 进程级模型缓存（按 spec path 缓存 `Arc<LlamaModel>`），或提供 `semq serve` 常驻模式 |

**快速 win**：Embedder/Reranker 的 Mutex 去掉或池化，能立即释放并行度。

---

## 5. 存储层（semquery-storage）

| 问题 | 影响 | 优化思路 |
|---|---|---|
| `SqliteStorage` 是全局 `Arc<Mutex<Connection>>` | 读写全串行 | 用连接池（r2d2+rusqlite），读并发，写单独一个连接 |
| 没设 SQLite 性能 pragmas | 默认 rollback journal 写放大严重 | `PRAGMA journal_mode=WAL`、`synchronous=NORMAL`、`cache_size`、可能 `mmap_size` ✅ |
| `chunks(doc_id)` 没建索引 | 删除文档时全表扫描 | `CREATE INDEX idx_chunk_documents_doc_id ON chunk_documents(doc_id)` ✅ |
| embedding 序列化逐 float `extend_from_slice` | 函数调用+拷贝开销 | 用 `bytemuck` 或 `unsafe { from_raw_parts }` 一次 memcpy |
| 每次 SQL 都重新 prepare statement | 热路径解析开销 | 用 `CachedStatement` 或持久化 prepared statements |
| `status` 用 `list_documents().len()` 计数 | 随文档数线性增长 | 加 `SELECT COUNT(*)` |

**快速 win**：WAL + cache_size + chunks doc_id 索引，改动很小但索引和查询都受益。

---

## 6. 架构/工程层面

- **`tracing` 可观测性**：现在只有 `--verbose` 打印耗时，难定位真实瓶颈。给 embed、search、rerank、LLM decode、storage query 加 span，后续优化才有数据。
- **错误类型太粗**：大量 `Other(String)`，没法区分可重试错误和致命错误，也限制了未来做重试/熔断。
- **并发控制**：以后要是有 `semq serve` 或多 collection 索引，需要 `Semaphore` 限流，避免 burst 把内存/线程打满。

---

## 建议优先级

1. **存储**：WAL + 单次事务 batch 写入 + `chunks(doc_id)` 索引
2. **检索**：BM25 和向量召回并行
3. **模型层**：Embedder/Reranker 去 Mutex 或池化
4. **问答**：上下文 token 预算 + 尾部截断
5. **索引器**：流式读取、批量去重、减少重复 tokenize
6. **长期**：`semq serve` 常驻模式 + 进程级模型缓存

这些改动基本都在现有架构内，不需要破坏 crate 分层。

---

## 7. 正确性与健壮性缺陷（P0，优先于性能优化）

> 与上面 1-6 节的性能方向不同，本节是**已确认的正确性 bug 或设计未实现**。它们会导致检索结果静默错误或直接崩溃，应优先于任何性能优化处理。

### 7.1 embedding 模型升级后旧向量静默错配 ✅ 已完成

- 改动：新增 `meta(key, value)` KV 表。`Indexer` 新增 `needs_reindex()` 同时检查 embedding spec 变化和 indexing 配置变化（chunk_size/chunk_overlap），任一不一致则强制全量重索引。索引完成后写入 `model_versions` + `meta["indexing"]`。

### 7.2 `SentenceSplitter` 的 `byte_range` 与原文偏移错位 ✅ 已完成

- 改动：重写 `split_paragraphs`/`split_sentences`/`split_words`/`split_chars` 返回 `(text_slice, byte_offset_in_original_text)` 二元组，所有偏移基于原文字节位置。`chunk()` 用 unit 的 `byte_offset` 直接计算 `byte_range`，`chunk_text` 从 `text[start..end]` 提取而非重新 join。新增 3 个测试验证 `text[byte_range] == chunk.text`（含多字节字符和段落场景）。

### 7.3 向量维度硬编码 512

- **位置**：`semquery-storage/src/sqlite.rs:196` `embedding FLOAT[512]`
- **现状**：schema 写死 `[512]`，但 `embedder.dimension()` 是动态的，且 `registry.rs` 已声明支持 `BGELargeZHV15`(1024) / `BGEM3`(1024)。
- **影响**：用 1024 维模型建表后插入向量直接失败。
- **建议**：`init()` 时根据 `embedder.dimension()` 建表，或在 config 中固定维度并在加载时校验。

### 7.4 错误类型全是 `Other(String)`，错误分类丢失

- **位置**：`semquery-core/src/error.rs` 全部子错误
- **现状**：`ParseError` / `StoreError` / `EmbedError` / `RetrieveError` / `SynthError` / `LlmError` / `ModelError` 都只有一个 `Other(String)` 变体，`#[from]` 转换无处附着，分类是"假分类"。
- **影响**：调用方无法按错误类型编程式处理（如区分"可重试的下载失败"与"致命配置错误"），限制了未来做重试/熔断。
- **建议**：为每类补充 1-2 个真实变体（如 `StoreError::Io`、`ModelError::Download`、`LlmError::ContextOverflow`）。

### 7.5 `bm25()` 分数语义注释误导 ✅ 已完成

- 改动：`search_text` 中将 FTS5 `bm25()` 返回的负分取反（`-raw`），使 `ScoreExplain.bm25_score` 遵循 "higher is better" 约定。

### 7.6 `hub.ensure` 无条件写 `model_versions`

- **位置**：`semquery-model/src/hub.rs:20`
- **现状**：`ensure` 每次都 `begin_tx` + `set_model_version`，即使模型没变。
- **影响**：每次加载模型都多一次事务写，且掩盖了 7.1 的对比逻辑缺失。
- **建议**：先 `get_model_version` 对比，相同则跳过写。

---

## 建议优先级（合并版）

1. **正确性（P0）**：接线 `model_versions` 重索引触发（7.1）→ 修复 `SentenceSplitter` 字节偏移（7.2）→ 向量维度去硬编码（7.3）
2. **存储**：WAL + 单次事务 batch 写入 + `chunks(doc_id)` 索引
3. **检索**：BM25 和向量召回并行
4. **模型层**：Embedder/Reranker 去 Mutex 或池化
5. **问答**：上下文 token 预算 + 尾部截断
6. **索引器**：流式读取、批量去重、减少重复 tokenize
7. **长期**：`semq serve` 常驻模式 + 进程级模型缓存

> 正确性缺陷（第 1 项）应优先于所有性能优化：它们会导致检索结果静默错误或直接崩溃，性能优化无法弥补。
