pub mod txt_reader;

#[cfg(feature = "pdf")]
pub mod pdf_reader;

#[cfg(feature = "docx")]
pub mod docx_reader;

pub use txt_reader::TextFileReader;

#[cfg(feature = "pdf")]
pub use pdf_reader::PdfReader;

#[cfg(feature = "docx")]
pub use docx_reader::DocxReader;
