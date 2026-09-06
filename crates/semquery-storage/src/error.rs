use semquery_core::StoreError;

pub(crate) fn map_rusqlite(e: rusqlite::Error) -> StoreError {
  StoreError::Sqlite(e.to_string())
}
