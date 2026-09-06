//! SQLite-backed [`Storage`] implementation.
//!
//! [`Storage`]: semquery_core::traits::Storage

mod error;
mod sqlite;

pub use sqlite::SqliteStorage;
