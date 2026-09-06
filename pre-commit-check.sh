#!/usr/bin/env bash
set -euo pipefail

# semquery pre-commit check script
# Usage: ./pre-commit-check.sh
#
# Runs the full set of checks required before committing (see AGENTS.md):
#   - cargo check --workspace        fastest compile check
#   - cargo test --workspace         all tests
#   - cargo fmt --all -- --check     formatting
#   - cargo clippy --all-features -- -D warnings   lints (warnings are errors)

GREEN='\033[0;32m'
RED='\033[0;31m'
RESET='\033[0m'

# Run a check, printing a colored header. `set -e` aborts on the first failure.
run() {
  local name="$1"
  shift
  echo -e "\n${GREEN}=== $name ===${RESET}"
  "$@"
}

run "cargo check --workspace" cargo check --workspace
run "cargo test --workspace" cargo test --workspace
run "cargo fmt --all -- --check" cargo fmt --all -- --check
run "cargo clippy --all-features -- -D warnings" cargo clippy --all-features -- -D warnings

echo -e "\n${GREEN}=== All checks passed ===${RESET}"
