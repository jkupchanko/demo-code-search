#!/usr/bin/env bash
#
# Build the corpus from a checkout of qdrant/qdrant and load it into Qdrant.
#
# The embedding happens inside the cluster, so nothing here installs a model.
# The slow parts are rust-analyzer's LSIF pass and the rust-parser container,
# which is where most of the wall clock goes.

set -e

QDRANT_PATH=$(realpath "$1")

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ROOT_PATH=$SCRIPT_PATH/..

export QDRANT_PATH

# Whole .rs files, for the file viewer.
python "$ROOT_PATH/indexer/prepare/files_to_json.py"

# Folding ranges -> code snippets.
rustup run stable rust-analyzer -v lsif "$QDRANT_PATH" > "$ROOT_PATH/data/index.lsif"
python "$ROOT_PATH/indexer/prepare/convert_lsif_index.py"

# Function and struct signatures with their docstrings.
docker run --rm -v "$QDRANT_PATH":/source qdrant/rust-parser ./rust_parser /source \
  > "$ROOT_PATH/data/structures.json"

# One pass, three collections, every vector computed by Qdrant Cloud Inference.
python "$ROOT_PATH/indexer/build.py" --source files --fresh
