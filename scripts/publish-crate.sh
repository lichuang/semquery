#!/usr/bin/env bash
# Publish a single workspace crate to crates.io, but only if its local version
# differs from the version already published.
#
# Usage:
#   scripts/publish-crate.sh semquery-core
#   scripts/publish-crate.sh semquery-model --dry-run
#   scripts/publish-crate.sh semquery --allow-dirty
#
# Dependency order for the first release (publish from bottom to top).
# Dev-dependencies are also resolved from crates.io during publish, so they
# must be published before the crate that references them.
#
#   1. semquery-core
#   2. semquery-storage        (dev-dependency of semquery-model)
#   3. semquery-model          (depends on semquery-core; dev-depends on semquery-storage)
#   4. semquery-indexer        (depends on semquery-core, semquery-model, semquery-storage)
#   5. semquery-retrieve       (depends on semquery-core, semquery-model, semquery-storage)
#   6. semquery-synth          (depends on semquery-core, semquery-model, semquery-retrieve;
#                         dev-depends on semquery-indexer, semquery-storage)
#   7. semquery                (depends on all of the above)

set -euo pipefail

CRATE=""
DRY_RUN=false
ALLOW_DIRTY=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --allow-dirty)
      ALLOW_DIRTY=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 <crate-name> [--dry-run] [--allow-dirty]"
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 <crate-name> [--dry-run] [--allow-dirty]" >&2
      exit 1
      ;;
    *)
      CRATE="$1"
      shift
      ;;
  esac
done

if [[ -z "$CRATE" ]]; then
  echo "Usage: $0 <crate-name> [--dry-run] [--allow-dirty]" >&2
  exit 1
fi

for tool in cargo curl jq; do
  if ! command -v "$tool" &> /dev/null; then
    echo "Error: required tool '$tool' is not installed" >&2
    exit 1
  fi
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Verify the crate is part of the workspace.
if ! cargo metadata --no-deps --format-version 1 \
     | jq -e --arg name "$CRATE" '.packages[] | select(.name == $name)' > /dev/null; then
  echo "Error: crate '$CRATE' is not a member of this workspace" >&2
  exit 1
fi

LOCAL_VERSION=$(cargo metadata --no-deps --format-version 1 \
  | jq -r --arg name "$CRATE" '.packages[] | select(.name == $name) | .version')

echo "crate:  $CRATE"
echo "local:  $LOCAL_VERSION"

REMOTE_VERSION=$(curl -sS \
  -H "User-Agent: semquery-publish-script" \
  "https://crates.io/api/v1/crates/$CRATE" 2>/dev/null \
  | jq -r '.crate.max_version // empty' 2>/dev/null || true)

if [[ -z "${REMOTE_VERSION:-}" ]]; then
  echo "remote: <not published yet>"
else
  echo "remote: $REMOTE_VERSION"
fi

if [[ -n "${REMOTE_VERSION:-}" && "$LOCAL_VERSION" == "$REMOTE_VERSION" ]]; then
  echo "Versions are identical. Skipping publish."
  exit 0
fi

PUBLISH_ARGS=(-p "$CRATE")

if $DRY_RUN; then
  PUBLISH_ARGS+=(--dry-run)
fi

if $ALLOW_DIRTY; then
  PUBLISH_ARGS+=(--allow-dirty)
fi

echo "Publishing $CRATE@$LOCAL_VERSION..."
cargo publish "${PUBLISH_ARGS[@]}"
