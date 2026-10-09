//! Well-known keys of the `meta` key/value table shared across crates.

/// The `ModelSpec` (as JSON) of the *last successful indexing run*, written
/// only by the indexer after a run completes — never by model loading — so
/// the reindex check cannot be fooled by a freshly-loaded model spec (issue #8).
pub const EMBEDDING_BASELINE_KEY: &str = "embedding_baseline";

/// The `"{chunk_size}:{chunk_overlap}"` line recorded with the baseline,
/// detecting chunking-config drift that invalidates stored chunks.
pub const INDEXING_CONFIG_KEY: &str = "indexing";
