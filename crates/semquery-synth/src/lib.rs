//! Citation-grounded answer synthesis over retrieved passages.

pub mod citation;
pub mod prompt;
pub mod synthesizer;

pub use synthesizer::{Synthesizer, SynthesizerConfig};
