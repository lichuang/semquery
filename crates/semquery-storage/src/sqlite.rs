use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, Once};

use chrono::{DateTime, Utc};
use rusqlite::ffi::sqlite3_auto_extension;
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};
use semquery_core::{Chunk, Collection, Document, ModelRole, ModelSpec, Result, Storage, StorageTx, StoreError};
use sqlite_vec::sqlite3_vec_init;

use crate::error::map_rusqlite;

static VEC_EXT_LOADED: Once = Once::new();

fn ensure_vec_extension() {
  VEC_EXT_LOADED.call_once(|| unsafe {
    sqlite3_auto_extension(Some(std::mem::transmute::<
      *const (),
      unsafe extern "C" fn(
        *mut rusqlite::ffi::sqlite3,
        *mut *mut i8,
        *const rusqlite::ffi::sqlite3_api_routines,
      ) -> i32,
    >(sqlite3_vec_init as *const ())));
  });
}

fn embedding_to_bytes(vec: &[f32]) -> Vec<u8> {
  let mut bytes = Vec::with_capacity(vec.len() * 4);
  for &f in vec {
    bytes.extend_from_slice(&f.to_le_bytes());
  }
  bytes
}

pub struct SqliteStorage {
  conn: Arc<Mutex<Connection>>,
}

const DB_FILE_NAME: &str = "semquery.db";

impl SqliteStorage {
  pub fn open(path: impl AsRef<Path>) -> Result<Self> {
    ensure_vec_extension();
    let conn = Connection::open(path).map_err(map_rusqlite)?;
    conn
      .execute_batch(
        "PRAGMA busy_timeout = 5000; \
         PRAGMA foreign_keys = ON; \
         PRAGMA journal_mode = WAL; \
         PRAGMA synchronous = NORMAL; \
         PRAGMA cache_size = -64000;",
      )
      .map_err(map_rusqlite)?;
    Ok(Self {
      conn: Arc::new(Mutex::new(conn)),
    })
  }

  pub fn open_in_memory() -> Result<Self> {
    ensure_vec_extension();
    let conn = Connection::open_in_memory().map_err(map_rusqlite)?;
    conn
      .execute_batch(
        "PRAGMA busy_timeout = 5000; \
       PRAGMA foreign_keys = ON; \
       PRAGMA synchronous = NORMAL; \
       PRAGMA cache_size = -64000;",
      )
      .map_err(map_rusqlite)?;
    Ok(Self {
      conn: Arc::new(Mutex::new(conn)),
    })
  }

  /// Open the default SQLite database inside a workspace directory.
  /// Keeps the concrete filename (`semquery.db`) encapsulated in the storage crate
  /// so callers don't assume SQLite implementation details.
  pub fn open_workspace(workspace: impl AsRef<Path>) -> Result<Self> {
    let path = workspace.as_ref().join(DB_FILE_NAME);
    Self::open(path)
  }

  fn init_vectors(&self, conn: &Connection, dimension: usize) -> Result<()> {
    let expected_type = format!("FLOAT[{dimension}]");

    let existing_sql: Option<String> = conn
      .query_row(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'",
        [],
        |r| r.get(0),
      )
      .optional()
      .map_err(map_rusqlite)?;

    if let Some(sql) = existing_sql {
      if !sql.contains(&expected_type) {
        return Err(
          StoreError::SchemaMismatch {
            expected: expected_type,
          }
          .into(),
        );
      }
      return Ok(());
    }

    let sql = format!(
      "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
         chunk_id  TEXT PRIMARY KEY,
         embedding {expected_type} distance_metric=cosine
       );"
    );
    conn.execute_batch(&sql).map_err(map_rusqlite)?;
    Ok(())
  }
}

fn poisoned() -> StoreError {
  StoreError::MutexPoisoned
}

fn insert_document(conn: &Connection, doc: &Document) -> rusqlite::Result<()> {
  conn.execute(
    "INSERT OR REPLACE INTO documents
     (doc_id, content_hash, content_size, indexed_at)
   VALUES (?1, ?2, ?3, ?4)",
    params![
      doc.id,
      doc.content_hash,
      doc.content_size as i64,
      doc.indexed_at.to_rfc3339(),
    ],
  )?;
  Ok(())
}

fn insert_chunks(conn: &Connection, chunks: &[Chunk]) -> rusqlite::Result<()> {
  let mut stmt = conn.prepare(
    "INSERT OR IGNORE INTO chunks (chunk_id, text, start_byte, end_byte)
     VALUES (?1, ?2, ?3, ?4)",
  )?;
  for chunk in chunks {
    stmt.execute(params![
      chunk.id,
      chunk.text,
      chunk.byte_range.start as i64,
      chunk.byte_range.end as i64,
    ])?;
  }
  Ok(())
}

fn insert_chunk_documents(conn: &Connection, chunk_ids: &[String], doc_id: &str) -> rusqlite::Result<()> {
  let mut stmt = conn.prepare("INSERT OR IGNORE INTO chunk_documents (chunk_id, doc_id) VALUES (?1, ?2)")?;
  for cid in chunk_ids {
    stmt.execute(params![cid, doc_id])?;
  }
  Ok(())
}

fn insert_vectors(conn: &Connection, chunk_ids: &[String], embeddings: &[Vec<f32>]) -> rusqlite::Result<()> {
  let mut stmt = conn.prepare("INSERT OR REPLACE INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)")?;
  for (id, emb) in chunk_ids.iter().zip(embeddings.iter()) {
    let bytes = embedding_to_bytes(emb);
    stmt.execute(params![id, bytes])?;
  }
  Ok(())
}

fn insert_fts(conn: &Connection, chunk_ids: &[String], tokenized_texts: &[String]) -> rusqlite::Result<()> {
  let mut stmt = conn.prepare("INSERT OR REPLACE INTO fts_chunks (chunk_id, text) VALUES (?1, ?2)")?;
  for (id, text) in chunk_ids.iter().zip(tokenized_texts.iter()) {
    stmt.execute(params![id, text])?;
  }
  Ok(())
}

fn insert_document_path(conn: &Connection, doc_id: &str, path: &str) -> rusqlite::Result<()> {
  conn.execute(
    "INSERT OR REPLACE INTO document_paths (doc_id, file_path) VALUES (?1, ?2)",
    params![doc_id, path],
  )?;
  Ok(())
}

fn delete_document(conn: &Connection, doc_id: &str) -> rusqlite::Result<()> {
  conn.execute("DELETE FROM document_paths WHERE doc_id = ?1", params![doc_id])?;
  conn.execute("DELETE FROM documents WHERE doc_id = ?1", params![doc_id])?;
  Ok(())
}

fn delete_chunks_by_doc(conn: &Connection, doc_id: &str) -> rusqlite::Result<()> {
  conn.execute("DELETE FROM chunk_documents WHERE doc_id = ?1", params![doc_id])?;

  let orphaned = "chunk_id NOT IN (SELECT chunk_id FROM chunk_documents)";
  conn.execute(
    &format!("DELETE FROM vec_chunks WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE {orphaned})"),
    [],
  )?;
  conn.execute(
    &format!("DELETE FROM fts_chunks WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE {orphaned})"),
    [],
  )?;
  conn.execute(&format!("DELETE FROM chunks WHERE {orphaned}"), [])?;
  Ok(())
}

fn set_model_version(conn: &Connection, role: ModelRole, version: &ModelSpec) -> rusqlite::Result<()> {
  conn.execute(
    "INSERT OR REPLACE INTO model_versions (role, repo_id, filename, revision, checksum)
     VALUES (?1, ?2, ?3, ?4, ?5)",
    params![
      role.as_str(),
      version.repo_id,
      version.filename,
      version.revision,
      version.checksum
    ],
  )?;
  Ok(())
}

impl Storage for SqliteStorage {
  fn init(&self, vector_dimension: usize) -> Result<()> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    conn
      .execute_batch(
        "CREATE TABLE IF NOT EXISTS documents (
         -- Indexed file; one row per .txt/.md added via `semq add`.
         -- `doc_id` is a stable identifier derived from the file path.
         doc_id       TEXT PRIMARY KEY,  -- stable document identifier
         content_hash TEXT NOT NULL,     -- SHA-256 of file bytes; drives incremental reindex
         content_size INTEGER NOT NULL,  -- byte length of the original file
         indexed_at   TEXT NOT NULL      -- RFC3339 UTC timestamp of last successful index
       );
       CREATE TABLE IF NOT EXISTS document_paths (
         -- Separates the mutable file system path from the stable document id.
         doc_id     TEXT PRIMARY KEY,
         file_path  TEXT NOT NULL UNIQUE,
         FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
       );
       CREATE TABLE IF NOT EXISTS chunks (
          -- Text block produced by the Chunker; `text` is the exact string embedded.
          -- chunk_id is the SHA-256 of `text`, so identical content is stored once.
          -- Parallel `vec_chunks` (sqlite-vec) and `fts_chunks` (FTS5) tables
          -- are keyed by the same chunk_id.
          -- doc_id is NOT stored here — the many-to-many mapping lives in
          -- `chunk_documents` so a shared chunk can belong to multiple files
          -- without one overwriting the other.
          chunk_id   TEXT PRIMARY KEY,  -- SHA-256 of `text`; content-addressed dedup
          text       TEXT NOT NULL,     -- full original text actually embedded
          start_byte INTEGER NOT NULL,  -- byte offset in the source file (inclusive)
          end_byte   INTEGER NOT NULL  -- byte offset in the source file (exclusive)
        );
        CREATE TABLE IF NOT EXISTS chunk_documents (
          -- Many-to-many junction: a chunk (identified by its text hash) can
          -- appear in multiple documents. This table records every (chunk, doc)
          -- pair so that deleting one document does not clobber the doc_id of
          -- a chunk still referenced by another document.
          chunk_id  TEXT NOT NULL,
          doc_id    TEXT NOT NULL,
          PRIMARY KEY (chunk_id, doc_id),
          FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
          FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        );
       CREATE TABLE IF NOT EXISTS model_versions (
         -- Records which model produced the current embeddings / rerank scores / chat answers.
         -- On embedding-model upgrade the stored vectors become stale; the indexer compares
         -- the stored spec here against the live one and triggers an explicit reindex
         -- rather than serving silently-mismatched vectors.
         role     TEXT PRIMARY KEY,   -- 'embedding' / 'reranker' / 'chat'
         repo_id  TEXT NOT NULL,      -- HuggingFace repo id, e.g. 'BAAI/bge-small-zh-v1.5'
         filename TEXT NOT NULL,      -- on-disk artifact filename (.onnx / .gguf)
         revision TEXT NOT NULL,      -- HuggingFace revision or commit hash
         checksum TEXT                -- optional content checksum for verifying downloads
       );
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
          -- FTS5 full-text index over jieba-pre-tokenised text.
          -- `text` here stores space-separated tokens (jieba output), NOT the raw chunk text.
          -- Raw text lives in `chunks.text`; this table only serves BM25 ranking.
          -- `unicode61` splits on the spaces between tokens, treating each jieba word as a term.
          chunk_id,
          text,
          tokenize='unicode61'
        );
        CREATE TABLE IF NOT EXISTS collections (
          name TEXT PRIMARY KEY,
          path TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (
          key   TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunk_documents_doc_id ON chunk_documents(doc_id);",
      )
      .map_err(map_rusqlite)?;
    if vector_dimension == 0 {
      return Ok(());
    }
    self.init_vectors(&conn, vector_dimension)
  }

  fn get_document(&self, doc_id: &str) -> Result<Option<Document>> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    let row = conn
      .query_row(
        "SELECT doc_id, content_hash, content_size, indexed_at
       FROM documents WHERE doc_id = ?1",
        params![doc_id],
        |r| {
          Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
            r.get::<_, String>(3)?,
          ))
        },
      )
      .optional()
      .map_err(map_rusqlite)?;

    match row {
      Some((id, content_hash, content_size, ts)) => {
        let indexed_at = DateTime::parse_from_rfc3339(&ts)
          .map_err(|e| StoreError::InvalidTimestamp(e.to_string()))?
          .with_timezone(&Utc);
        Ok(Some(Document {
          id,
          content_hash,
          content_size: content_size as usize,
          indexed_at,
        }))
      }
      None => Ok(None),
    }
  }

  fn list_documents(&self) -> Result<Vec<Document>> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    let mut stmt = conn
      .prepare("SELECT doc_id, content_hash, content_size, indexed_at FROM documents")
      .map_err(map_rusqlite)?;

    let rows: Vec<(String, String, i64, String)> = stmt
      .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
      .map_err(map_rusqlite)?
      .collect::<rusqlite::Result<Vec<_>>>()
      .map_err(map_rusqlite)?;

    let docs = rows
      .into_iter()
      .map(|(id, content_hash, content_size, ts)| {
        let indexed_at = DateTime::parse_from_rfc3339(&ts)
          .map_err(|e| StoreError::InvalidTimestamp(e.to_string()))?
          .with_timezone(&Utc);
        Ok(Document {
          id,
          content_hash,
          content_size: content_size as usize,
          indexed_at,
        })
      })
      .collect::<Result<Vec<_>>>()?;
    Ok(docs)
  }

  fn get_document_paths(&self, doc_ids: &[String]) -> Result<HashMap<String, String>> {
    if doc_ids.is_empty() {
      return Ok(HashMap::new());
    }
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    let placeholders = (0..doc_ids.len()).map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
      "SELECT doc_id, file_path FROM document_paths WHERE doc_id IN ({})",
      placeholders
    );
    let mut stmt = conn.prepare(&sql).map_err(map_rusqlite)?;
    let rows = stmt
      .query_map(params_from_iter(doc_ids.iter()), |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
      })
      .map_err(map_rusqlite)?
      .collect::<rusqlite::Result<Vec<_>>>()
      .map_err(map_rusqlite)?;
    Ok(rows.into_iter().collect())
  }

  fn get_chunks(&self, chunk_ids: &[String]) -> Result<Vec<Chunk>> {
    if chunk_ids.is_empty() {
      return Ok(Vec::new());
    }
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    let placeholders = (0..chunk_ids.len()).map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
      "SELECT c.chunk_id, (
         SELECT cd.doc_id FROM chunk_documents cd
         WHERE cd.chunk_id = c.chunk_id
         ORDER BY cd.doc_id LIMIT 1
       ) AS doc_id, c.text, c.start_byte, c.end_byte
       FROM chunks c
       WHERE c.chunk_id IN ({})",
      placeholders
    );
    let mut stmt = conn.prepare(&sql).map_err(map_rusqlite)?;
    let rows = stmt
      .query_map(params_from_iter(chunk_ids.iter()), |r| {
        Ok((
          r.get::<_, String>(0)?,
          r.get::<_, Option<String>>(1)?,
          r.get::<_, String>(2)?,
          r.get::<_, i64>(3)?,
          r.get::<_, i64>(4)?,
        ))
      })
      .map_err(map_rusqlite)?
      .collect::<rusqlite::Result<Vec<_>>>()
      .map_err(map_rusqlite)?;

    let chunks = rows
      .into_iter()
      .map(|(id, doc_id, text, start, end)| Chunk {
        id,
        doc_id: doc_id.unwrap_or_default(),
        text,
        byte_range: start as usize..end as usize,
      })
      .collect();
    Ok(chunks)
  }

  fn search_vectors(&self, embedding: &[f32], top_k: usize) -> Result<Vec<(String, f32)>> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    let bytes = embedding_to_bytes(embedding);
    let mut stmt = conn
      .prepare(
        "SELECT chunk_id, distance
         FROM vec_chunks
         WHERE embedding MATCH ?1 AND k = ?2
         ORDER BY distance",
      )
      .map_err(map_rusqlite)?;
    // sqlite-vec returns cosine distance (0 = identical, 2 = opposite).
    // Convert to similarity (1.0 = identical, -1.0 = opposite) so all
    // channels follow the "higher is better" convention uniformly.
    let rows = stmt
      .query_map(params![bytes, top_k as i64], |r| {
        let distance: f32 = r.get(1)?;
        Ok((r.get::<_, String>(0)?, 1.0 - distance))
      })
      .map_err(map_rusqlite)?
      .collect::<rusqlite::Result<Vec<_>>>()
      .map_err(map_rusqlite)?;
    Ok(rows)
  }

  fn search_text(&self, query: &str, top_k: usize) -> Result<Vec<(String, f32)>> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    let mut stmt = conn
      .prepare(
        "SELECT chunk_id, bm25(fts_chunks) AS score
         FROM fts_chunks
         WHERE fts_chunks MATCH ?1
         ORDER BY rank
         LIMIT ?2",
      )
      .map_err(map_rusqlite)?;
    // FTS5 bm25() returns negative scores (0 = best, more negative = worse).
    // Negate so higher = more relevant, matching the convention of all other channels.
    let rows = stmt
      .query_map(params![query, top_k as i64], |r| {
        let raw: f32 = r.get(1)?;
        Ok((r.get::<_, String>(0)?, -raw))
      })
      .map_err(map_rusqlite)?
      .collect::<rusqlite::Result<Vec<_>>>()
      .map_err(map_rusqlite)?;
    Ok(rows)
  }

  fn get_model_version(&self, role: ModelRole) -> Result<Option<ModelSpec>> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    let row = conn
      .query_row(
        "SELECT repo_id, filename, revision, checksum FROM model_versions WHERE role = ?1",
        params![role.as_str()],
        |r| {
          Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, Option<String>>(3)?,
          ))
        },
      )
      .optional()
      .map_err(map_rusqlite)?;

    match row {
      Some((repo_id, filename, revision, checksum)) => Ok(Some(ModelSpec {
        role,
        repo_id,
        filename,
        revision,
        checksum,
      })),
      None => Ok(None),
    }
  }

  fn list_collections(&self) -> Result<Vec<Collection>> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    let mut stmt = conn.prepare("SELECT name, path FROM collections").map_err(map_rusqlite)?;
    let rows = stmt
      .query_map([], |r| {
        Ok(Collection {
          name: r.get(0)?,
          path: r.get::<_, String>(1)?.into(),
        })
      })
      .map_err(map_rusqlite)?
      .collect::<rusqlite::Result<Vec<_>>>()
      .map_err(map_rusqlite)?;
    Ok(rows)
  }

  fn count_chunks(&self) -> Result<usize> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0)).map_err(map_rusqlite)?;
    Ok(count as usize)
  }

  fn set_model_version_atomic(&self, role: ModelRole, version: &ModelSpec) -> Result<()> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    set_model_version(&conn, role, version).map_err(map_rusqlite)?;
    Ok(())
  }

  fn get_meta(&self, key: &str) -> Result<Option<String>> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    let value: Option<String> = conn
      .query_row("SELECT value FROM meta WHERE key = ?1", params![key], |r| r.get(0))
      .optional()
      .map_err(map_rusqlite)?;
    Ok(value)
  }

  fn set_meta_atomic(&self, key: &str, value: &str) -> Result<()> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    conn
      .execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
        params![key, value],
      )
      .map_err(map_rusqlite)?;
    Ok(())
  }

  fn begin_tx(&self) -> Result<Box<dyn StorageTx + '_>> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    conn.execute_batch("BEGIN").map_err(map_rusqlite)?;
    Ok(Box::new(SqliteTransaction {
      conn: self.conn.clone(),
      committed: false,
    }))
  }
}

pub struct SqliteTransaction {
  conn: Arc<Mutex<Connection>>,
  committed: bool,
}

impl StorageTx for SqliteTransaction {
  fn add_document(&mut self, doc: &Document) -> Result<()> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    insert_document(&conn, doc).map_err(map_rusqlite)?;
    Ok(())
  }

  fn set_document_path(&mut self, doc_id: &str, path: &str) -> Result<()> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    insert_document_path(&conn, doc_id, path).map_err(map_rusqlite)?;
    Ok(())
  }

  fn delete_document(&mut self, doc_id: &str) -> Result<()> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    crate::sqlite::delete_document(&conn, doc_id).map_err(map_rusqlite)?;
    Ok(())
  }

  fn add_chunks(&mut self, chunks: &[Chunk]) -> Result<()> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    insert_chunks(&conn, chunks).map_err(map_rusqlite)?;
    Ok(())
  }

  fn add_chunk_documents(&mut self, chunk_ids: &[String], doc_id: &str) -> Result<()> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    insert_chunk_documents(&conn, chunk_ids, doc_id).map_err(map_rusqlite)?;
    Ok(())
  }

  fn delete_chunks_by_doc(&mut self, doc_id: &str) -> Result<()> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    crate::sqlite::delete_chunks_by_doc(&conn, doc_id).map_err(map_rusqlite)?;
    Ok(())
  }

  fn add_vectors(&mut self, chunk_ids: &[String], embeddings: &[Vec<f32>]) -> Result<()> {
    if chunk_ids.len() != embeddings.len() {
      return Err(
        StoreError::ArgumentMismatch {
          what: "chunk_ids and embeddings".into(),
          a: chunk_ids.len(),
          b: embeddings.len(),
        }
        .into(),
      );
    }
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    insert_vectors(&conn, chunk_ids, embeddings).map_err(map_rusqlite)?;
    Ok(())
  }

  fn add_fts_chunks(&mut self, chunk_ids: &[String], tokenized_texts: &[String]) -> Result<()> {
    if chunk_ids.len() != tokenized_texts.len() {
      return Err(
        StoreError::ArgumentMismatch {
          what: "chunk_ids and tokenized_texts".into(),
          a: chunk_ids.len(),
          b: tokenized_texts.len(),
        }
        .into(),
      );
    }
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    insert_fts(&conn, chunk_ids, tokenized_texts).map_err(map_rusqlite)?;
    Ok(())
  }

  fn set_model_version(&mut self, role: ModelRole, version: &ModelSpec) -> Result<()> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    crate::sqlite::set_model_version(&conn, role, version).map_err(map_rusqlite)?;
    Ok(())
  }

  fn add_collection(&mut self, name: &str, path: &str) -> Result<()> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    conn
      .execute(
        "INSERT OR REPLACE INTO collections (name, path) VALUES (?1, ?2)",
        params![name, path],
      )
      .map_err(map_rusqlite)?;
    Ok(())
  }

  fn delete_collection(&mut self, name: &str) -> Result<()> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    conn.execute("DELETE FROM collections WHERE name = ?1", params![name]).map_err(map_rusqlite)?;
    Ok(())
  }

  fn clear_collections(&mut self) -> Result<()> {
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    conn.execute("DELETE FROM collections", []).map_err(map_rusqlite)?;
    Ok(())
  }

  fn commit(&mut self) -> Result<()> {
    if self.committed {
      return Err(StoreError::TransactionAlreadyCommitted.into());
    }
    let conn = self.conn.lock().map_err(|_| poisoned())?;
    conn.execute_batch("COMMIT").map_err(map_rusqlite)?;
    self.committed = true;
    Ok(())
  }
}

impl Drop for SqliteTransaction {
  fn drop(&mut self) {
    if !self.committed
      && let Ok(conn) = self.conn.lock()
    {
      let _ = conn.execute_batch("ROLLBACK");
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use chrono::Utc;
  use semquery_core::{Chunk, Document, Storage, StorageTx};

  fn make_doc(id: &str, content: &str) -> Document {
    Document {
      id: id.to_string(),
      content_hash: format!("hash-{content}"),
      content_size: content.len(),
      indexed_at: Utc::now(),
    }
  }

  fn make_chunk(id: &str, doc_id: &str, text: &str, start: usize, end: usize) -> Chunk {
    Chunk {
      id: id.to_string(),
      doc_id: doc_id.to_string(),
      text: text.to_string(),
      byte_range: start..end,
    }
  }

  fn commit<F: FnOnce(&mut dyn StorageTx)>(storage: &SqliteStorage, body: F) {
    let mut tx = storage.begin_tx().unwrap();
    body(&mut *tx);
    tx.commit().unwrap();
  }

  #[test]
  fn test_document_crud() {
    let storage = SqliteStorage::open_in_memory().unwrap();
    storage.init(512).unwrap();

    let doc1 = make_doc("doc1.txt", "hello world");
    let doc2 = make_doc("doc2.txt", "foo bar");
    commit(&storage, |tx| {
      tx.add_document(&doc1).unwrap();
      tx.add_document(&doc2).unwrap();
    });

    let got = storage.get_document("doc1.txt").unwrap();
    assert!(got.is_some());
    assert_eq!(got.unwrap().content_hash, "hash-hello world");

    let list = storage.list_documents().unwrap();
    assert_eq!(list.len(), 2);

    commit(&storage, |tx| {
      tx.delete_document("doc1.txt").unwrap();
    });
    assert!(storage.get_document("doc1.txt").unwrap().is_none());
    assert_eq!(storage.list_documents().unwrap().len(), 1);
  }

  #[test]
  fn test_chunk_crud() {
    let storage = SqliteStorage::open_in_memory().unwrap();
    storage.init(512).unwrap();

    let doc = make_doc("doc1.txt", "hello world");
    let c1 = make_chunk("c1", "doc1.txt", "hello", 0, 5);
    let c2 = make_chunk("c2", "doc1.txt", "world", 6, 11);
    commit(&storage, |tx| {
      tx.add_document(&doc).unwrap();
      tx.add_chunks(&[c1, c2]).unwrap();
      tx.add_chunk_documents(&["c1".to_string(), "c2".to_string()], "doc1.txt").unwrap();
    });

    let got = storage.get_chunks(&["c1".to_string(), "c2".to_string()]).unwrap();
    assert_eq!(got.len(), 2);

    commit(&storage, |tx| {
      tx.delete_chunks_by_doc("doc1.txt").unwrap();
    });
    let gone = storage.get_chunks(&["c1".to_string()]).unwrap();
    assert!(gone.is_empty());
  }

  #[test]
  fn test_shared_chunk_survives_deleting_one_doc() {
    let storage = SqliteStorage::open_in_memory().unwrap();
    storage.init(512).unwrap();

    let doc_a = make_doc("doc_a.txt", "shared text");
    let doc_b = make_doc("doc_b.txt", "shared text");
    let shared_chunk = make_chunk("shared", "doc_a.txt", "shared text", 0, 11);

    commit(&storage, |tx| {
      tx.add_document(&doc_a).unwrap();
      tx.add_chunks(&[shared_chunk]).unwrap();
      tx.add_chunk_documents(&["shared".to_string()], "doc_a.txt").unwrap();
    });
    commit(&storage, |tx| {
      tx.add_document(&doc_b).unwrap();
      tx.add_chunks(&[make_chunk("shared", "doc_b.txt", "shared text", 0, 11)]).unwrap();
      tx.add_chunk_documents(&["shared".to_string()], "doc_b.txt").unwrap();
    });

    let got = storage.get_chunks(&["shared".to_string()]).unwrap();
    assert_eq!(got.len(), 1);
    assert!(!got[0].text.is_empty());

    commit(&storage, |tx| {
      tx.delete_chunks_by_doc("doc_a.txt").unwrap();
    });

    let got = storage.get_chunks(&["shared".to_string()]).unwrap();
    assert_eq!(got.len(), 1, "shared chunk must survive when doc_b still references it");
  }

  #[test]
  fn test_delete_chunks_cascades_vectors_and_fts() {
    let storage = SqliteStorage::open_in_memory().unwrap();
    storage.init(512).unwrap();

    let doc = make_doc("doc1.txt", "content");
    let chunk = make_chunk("c1", "doc1.txt", "hello", 0, 5);
    let embedding = vec![0.1_f32; 512];
    let tokenized = "hello".to_string();

    commit(&storage, |tx| {
      tx.add_document(&doc).unwrap();
      tx.add_chunks(&[chunk]).unwrap();
      tx.add_chunk_documents(&["c1".to_string()], "doc1.txt").unwrap();
      tx.add_vectors(&["c1".to_string()], &[embedding]).unwrap();
      tx.add_fts_chunks(&["c1".to_string()], &[tokenized]).unwrap();
    });

    assert!(!storage.search_vectors(&vec![0.1_f32; 512], 10).unwrap().is_empty());
    assert!(!storage.search_text("hello", 10).unwrap().is_empty());

    commit(&storage, |tx| {
      tx.delete_chunks_by_doc("doc1.txt").unwrap();
    });

    assert!(storage.get_chunks(&["c1".to_string()]).unwrap().is_empty());
    assert!(storage.search_vectors(&vec![0.1_f32; 512], 10).unwrap().is_empty());
    assert!(storage.search_text("hello", 10).unwrap().is_empty());
  }

  #[test]
  fn test_model_version() {
    let storage = SqliteStorage::open_in_memory().unwrap();
    storage.init(512).unwrap();

    assert!(storage.get_model_version(ModelRole::Embedding).unwrap().is_none());

    let spec = ModelSpec {
      role: ModelRole::Embedding,
      repo_id: "BAAI/bge-small-zh-v1.5".into(),
      filename: "model.onnx".into(),
      revision: "main".into(),
      checksum: Some("abc123".into()),
    };
    commit(&storage, |tx| {
      tx.set_model_version(ModelRole::Embedding, &spec).unwrap();
    });

    let got = storage.get_model_version(ModelRole::Embedding).unwrap().unwrap();
    assert_eq!(got.repo_id, "BAAI/bge-small-zh-v1.5");
    assert_eq!(got.checksum.as_deref(), Some("abc123"));
  }

  #[test]
  fn test_vector_search() {
    let storage = SqliteStorage::open_in_memory().unwrap();
    storage.init(512).unwrap();

    let doc = make_doc("doc1.txt", "content");
    let chunk = make_chunk("c0", "doc1.txt", "text", 0, 4);

    let base = vec![0.0_f32; 512];
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    let mut ids: Vec<String> = Vec::new();
    let mut chunks = vec![chunk];
    for i in 0..10 {
      let mut v = base.clone();
      v[i] = 1.0;
      vectors.push(v);
      ids.push(format!("c{i}"));
      let ch = make_chunk(&format!("c{i}"), "doc1.txt", "text", i, i + 1);
      chunks.push(ch);
    }
    commit(&storage, |tx| {
      tx.add_document(&doc).unwrap();
      tx.add_chunks(&chunks).unwrap();
      tx.add_chunk_documents(&ids, "doc1.txt").unwrap();
      tx.add_vectors(&ids, &vectors).unwrap();
    });

    let mut query = vec![0.0_f32; 512];
    query[3] = 1.0;
    let hits = storage.search_vectors(&query, 3).unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].0, "c3");
    assert!(hits[0].1 > hits[1].1);
  }

  #[test]
  fn test_text_search() {
    let storage = SqliteStorage::open_in_memory().unwrap();
    storage.init(512).unwrap();

    let doc = make_doc("doc1.txt", "content");
    let chunks = [
      ("c1", "分布式 共识 算法 解决 多个 节点 达成 一致 的 问题"),
      ("c2", "Raft 是 一种 易于 理解 的 共识 算法"),
      ("c3", "今天 天气 不错"),
    ];
    let chunk_objs: Vec<Chunk> =
      chunks.iter().map(|(id, text)| make_chunk(id, "doc1.txt", text, 0, text.len())).collect();
    let ids: Vec<String> = chunks.iter().map(|(id, _)| id.to_string()).collect();
    let texts: Vec<String> = chunks.iter().map(|(_, text)| text.to_string()).collect();
    commit(&storage, |tx| {
      tx.add_document(&doc).unwrap();
      tx.add_chunks(&chunk_objs).unwrap();
      tx.add_chunk_documents(&ids, "doc1.txt").unwrap();
      tx.add_fts_chunks(&ids, &texts).unwrap();
    });

    let hits = storage.search_text("共识 算法", 10).unwrap();
    assert_eq!(hits.len(), 2);
    let ids: Vec<&str> = hits.iter().map(|(id, _)| id.as_str()).collect();
    assert!(ids.contains(&"c1"));
    assert!(ids.contains(&"c2"));
    assert!(!ids.contains(&"c3"));
  }

  #[test]
  fn test_transaction_commit() {
    let storage = SqliteStorage::open_in_memory().unwrap();
    storage.init(512).unwrap();

    let doc = make_doc("doc1.txt", "content");
    let chunk = make_chunk("c1", "doc1.txt", "hello", 0, 5);
    let tokenized = "hello".to_string();
    let embedding = vec![0.1_f32; 512];

    {
      let mut tx = storage.begin_tx().unwrap();
      tx.add_document(&doc).unwrap();
      tx.add_chunks(std::slice::from_ref(&chunk)).unwrap();
      tx.add_chunk_documents(&["c1".to_string()], "doc1.txt").unwrap();
      tx.add_vectors(&["c1".to_string()], std::slice::from_ref(&embedding)).unwrap();
      tx.add_fts_chunks(&["c1".to_string()], std::slice::from_ref(&tokenized)).unwrap();
      tx.commit().unwrap();
    }

    assert!(storage.get_document("doc1.txt").unwrap().is_some());
    assert_eq!(storage.get_chunks(&["c1".to_string()]).unwrap().len(), 1);
    let hits = storage.search_vectors(&embedding, 1).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, "c1");
    assert_eq!(storage.search_text("hello", 10).unwrap().len(), 1);
  }

  #[test]
  fn test_transaction_rollback_on_drop() {
    let storage = SqliteStorage::open_in_memory().unwrap();
    storage.init(512).unwrap();

    let doc = make_doc("doc1.txt", "content");

    {
      let mut tx = storage.begin_tx().unwrap();
      tx.add_document(&doc).unwrap();
    }

    assert!(
      storage.get_document("doc1.txt").unwrap().is_none(),
      "doc must not persist without commit"
    );
  }

  #[test]
  fn test_transaction_commit_after_failure() {
    let storage = SqliteStorage::open_in_memory().unwrap();
    storage.init(512).unwrap();

    let doc = make_doc("doc1.txt", "content");
    let chunk = make_chunk("c1", "doc1.txt", "hello", 0, 5);

    let result = {
      let mut tx = storage.begin_tx().unwrap();
      tx.add_document(&doc).unwrap();
      tx.add_chunks(&[chunk]).unwrap();
      tx.add_vectors(&["c1".to_string(), "c2".to_string()], &[vec![0.0_f32; 512]]).unwrap_err();
      Err::<(), _>(StoreError::Sqlite("forced".into()))
    };
    assert!(result.is_err());

    assert!(
      storage.get_document("doc1.txt").unwrap().is_none(),
      "partial writes must be rolled back"
    );
    assert!(storage.get_chunks(&["c1".to_string()]).unwrap().is_empty());
  }

  #[test]
  fn test_concurrent_reads() {
    let storage = Arc::new(SqliteStorage::open_in_memory().unwrap());
    storage.init(512).unwrap();

    let doc = make_doc("doc1.txt", "concurrent read test");
    let chunk = make_chunk("c1", "doc1.txt", "hello world", 0, 11);
    commit(&storage, |tx| {
      tx.add_document(&doc).unwrap();
      tx.add_chunks(&[chunk]).unwrap();
      tx.add_chunk_documents(&["c1".to_string()], "doc1.txt").unwrap();
      tx.add_vectors(&["c1".to_string()], &[vec![0.1_f32; 512]]).unwrap();
      tx.add_fts_chunks(&["c1".to_string()], &["hello world".to_string()]).unwrap();
    });

    let handles: Vec<_> = (0..8)
      .map(|_| {
        let s = storage.clone();
        std::thread::spawn(move || {
          assert!(s.get_document("doc1.txt").unwrap().is_some());
          assert!(!s.get_chunks(&["c1".to_string()]).unwrap().is_empty());
          assert!(!s.search_vectors(&vec![0.1_f32; 512], 10).unwrap().is_empty());
          assert!(!s.search_text("hello", 10).unwrap().is_empty());
          assert_eq!(s.count_chunks().unwrap(), 1);
        })
      })
      .collect();

    for h in handles {
      h.join().unwrap();
    }
  }

  #[test]
  fn test_concurrent_reads_and_writes() {
    let storage = Arc::new(SqliteStorage::open_in_memory().unwrap());
    storage.init(512).unwrap();

    for i in 0..5 {
      let doc = make_doc(&format!("doc{i}.txt"), &format!("content {i}"));
      let chunk = make_chunk(&format!("c{i}"), &format!("doc{i}.txt"), &format!("text {i}"), 0, 6);
      commit(&storage, |tx| {
        tx.add_document(&doc).unwrap();
        tx.add_chunks(&[chunk]).unwrap();
        tx.add_chunk_documents(&[format!("c{i}")], &format!("doc{i}.txt")).unwrap();
      });
    }

    let reader = storage.clone();
    let read_handle = std::thread::spawn(move || {
      for _ in 0..50 {
        let docs = reader.list_documents().unwrap();
        assert!(docs.len() >= 5);
        let _ = reader.count_chunks().unwrap();
      }
    });

    let writer = storage.clone();
    let write_handle = std::thread::spawn(move || {
      for i in 0..5 {
        let doc = make_doc(&format!("extra{i}.txt"), &format!("extra {i}"));
        let chunk = make_chunk(
          &format!("extra_c{i}"),
          &format!("extra{i}.txt"),
          &format!("extra {i}"),
          0,
          7,
        );
        commit(&writer, |tx| {
          tx.add_document(&doc).unwrap();
          tx.add_chunks(&[chunk]).unwrap();
          tx.add_chunk_documents(&[format!("extra_c{i}")], &format!("extra{i}.txt")).unwrap();
        });
      }
    });

    read_handle.join().unwrap();
    write_handle.join().unwrap();

    let final_count = storage.list_documents().unwrap().len();
    assert_eq!(final_count, 10);
  }
}
