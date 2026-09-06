//! Library facade for semquery.
//!
//! This crate exposes the `Engine` and configuration types so that downstream
//! projects (e.g. `semquery-proc`) can embed semquery as a library, while the CLI
//! binary continues to live in `src/main.rs`.

pub mod config;
pub mod engine;

// Re-export the core crate so library consumers can access shared types
// (Answer, Citation, Verbose, etc.) without adding a separate dependency.
pub use semquery_core;

pub use config::DocqConfig;
pub use engine::{Engine, EngineComponents, EngineConfig};
