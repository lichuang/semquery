//! Citation marker extraction from LLM-generated answers.

use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

static CITATION_RE: OnceLock<Regex> = OnceLock::new();

fn citation_re() -> &'static Regex {
  CITATION_RE.get_or_init(|| Regex::new(r"\[(\d+)\]").unwrap())
}

/// Extract valid citation markers (`[1]`, `[2]`, ...) from an LLM answer.
///
/// The LLM is instructed to cite sources as `[N]` markers, where N maps to
/// the N-th retrieved chunk. This function:
///
/// 1. Scans `answer` for all `[N]` patterns via regex.
/// 2. Keeps only markers that exist in `valid_markers` (the set of markers
///    the Synthesizer actually provided as context — e.g. `[1]`..`[5]` when
///    5 chunks were retrieved). Markers like `[6]` or `[0]` are dropped.
/// 3. Deduplicates: a marker appearing twice in the answer is kept once.
///
/// Returns the valid markers in order of first appearance. The caller
/// (`Synthesizer::ask`) then back-fills each marker with the corresponding
/// chunk's `doc_id` and `byte_range` to produce a `Citation`.
pub fn parse_citations(answer: &str, valid_markers: &[String]) -> Vec<String> {
  let valid: HashSet<&str> = valid_markers.iter().map(|s| s.as_str()).collect();
  let mut seen = HashSet::new();
  let mut result = Vec::new();

  for cap in citation_re().captures_iter(answer) {
    let marker = cap.get(0).unwrap().as_str();
    if !valid.contains(marker) || !seen.insert(marker.to_string()) {
      continue;
    }
    result.push(marker.to_string());
  }

  result
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_parse_citations_basic() {
    let answer = "The answer is [1] because [2] says so.";
    let valid = vec!["[1]".to_string(), "[2]".to_string()];
    let result = parse_citations(answer, &valid);
    assert_eq!(result, vec!["[1]", "[2]"]);
  }

  #[test]
  fn test_parse_citations_filters_invalid() {
    let answer = "See [1] and [5] for details.";
    let valid = vec!["[1]".to_string(), "[2]".to_string()];
    let result = parse_citations(answer, &valid);
    assert_eq!(result, vec!["[1]"]);
  }

  #[test]
  fn test_parse_citations_dedup() {
    let answer = "As mentioned in [1], see [1] again.";
    let valid = vec!["[1]".to_string()];
    let result = parse_citations(answer, &valid);
    assert_eq!(result, vec!["[1]"]);
  }

  #[test]
  fn test_parse_citations_no_match() {
    let answer = "No citations here.";
    let valid = vec!["[1]".to_string()];
    let result = parse_citations(answer, &valid);
    assert!(result.is_empty());
  }
}
