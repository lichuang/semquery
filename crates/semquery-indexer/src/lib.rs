//! File reading, chunking, and indexing pipeline.

pub mod chunker;
mod indexer;
pub mod reader;
pub mod reader_registry;
pub mod tokenizer;

pub use chunker::SentenceSplitter;
#[cfg(feature = "docx")]
pub use reader::DocxReader;
#[cfg(feature = "pdf")]
pub use reader::PdfReader;
pub use reader::TextFileReader;
pub use reader_registry::ReaderRegistry;
pub use tokenizer::{JiebaSegmenter, jieba_tokenize};

pub use indexer::{IndexStats, Indexer, IndexerConfig, collect_index_stats};
