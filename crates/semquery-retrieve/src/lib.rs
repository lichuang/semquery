//! BM25 + vector recall, RRF fusion, and cross-encoder rerank.

pub mod fusion;
pub mod retriever;

pub use retriever::{Retriever, RetrieverConfig};
