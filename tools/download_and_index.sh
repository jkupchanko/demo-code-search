#!/usr/bin/env bash

set -e

# Ensure current path is project root
cd "$(dirname "$0")/../"

git clone https://github.com/qdrant/qdrant.git /tmp/qdrant

INDEXED_COMMIT=$(git -C /tmp/qdrant rev-parse HEAD)

bash -x tools/index_qdrant.sh /tmp/qdrant

rm -rf /tmp/qdrant

# Result links carry line numbers computed from this commit, so the API has to
# resolve them against it rather than against a branch that keeps moving.
# Printed last so it is the final thing in the log, where it is easy to find.
echo
echo "=============================================================="
echo "Indexed qdrant/qdrant at commit: $INDEXED_COMMIT"
echo "Set INDEXED_COMMIT to this value in the Vercel project so"
echo "result links point at the code that was actually indexed."
echo "=============================================================="
