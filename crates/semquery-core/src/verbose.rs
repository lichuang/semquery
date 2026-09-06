//! Lightweight verbose-progress helper for timing multi-step operations.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Tracks whether the configured logger is already duplicating log output to
/// the terminal. When it is not, `Verbose` falls back to `eprintln!` so that
/// progress messages are always visible to the user.
static LOGS_TO_TERMINAL: OnceLock<AtomicBool> = OnceLock::new();

/// Tell `Verbose` whether log records are already being printed to the terminal.
/// This should be called once by the application after the logger is initialized.
pub fn set_log_terminal_output(enabled: bool) {
  let _ = LOGS_TO_TERMINAL.set(AtomicBool::new(enabled));
}

fn logs_to_terminal() -> bool {
  LOGS_TO_TERMINAL.get().map(|b| b.load(Ordering::Relaxed)).unwrap_or(false)
}

/// Emit a verbose progress message.
///
/// The raw message is written to the log file under the fixed `semquery` target so
/// that it appears as `INFO [semquery] <msg>` rather than exposing the internal
/// `semquery_core::verbose` module path. If the logger is not already duplicating
/// output to the terminal, the message is also prefixed with `[semquery]` and
/// printed to stderr.
fn emit_verbose(msg: &str) {
  log::info!(target: "semquery", "{}", msg);
  if !logs_to_terminal() {
    eprintln!("[semquery] {msg}");
  }
}

/// On/off flag for verbose progress output.
#[derive(Clone, Copy, Debug, Default)]
pub struct Verbose(pub bool);

impl Verbose {
  pub fn enabled(&self) -> bool {
    self.0
  }

  /// Print an informational message when verbose mode is on.
  /// The message is always written to the log file and printed to the terminal
  /// unless the logger is already duplicating output to the terminal.
  pub fn log(&self, msg: &str) {
    if self.0 {
      emit_verbose(msg);
    }
  }

  /// Print the elapsed time for a completed step.
  pub fn step(&self, name: &str, elapsed: Duration) {
    if self.0 {
      emit_verbose(&format!("{name}: {} ms", elapsed.as_millis()));
    }
  }

  /// Start a timed step; the time is printed when the returned `Step` is dropped.
  pub fn start(&self, name: &'static str) -> Step {
    Step::new(*self, name)
  }
}

/// RAII timer for a single verbose step.
pub struct Step {
  verbose: Verbose,
  name: &'static str,
  start: Instant,
}

impl Step {
  pub fn new(verbose: Verbose, name: &'static str) -> Self {
    if verbose.0 {
      emit_verbose(&format!("{name}..."));
    }
    Self {
      verbose,
      name,
      start: Instant::now(),
    }
  }
}

impl Drop for Step {
  fn drop(&mut self) {
    self.verbose.step(self.name, self.start.elapsed());
  }
}
