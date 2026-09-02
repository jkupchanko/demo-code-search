"""Build the three collections, with the cluster doing every embedding.

    python indexer/build.py --source qdrant          # copy from the old demo's collections
    python indexer/build.py --source files           # from data/*.json (tools/index_qdrant.sh)
    python indexer/build.py --only signatures        # one collection at a time
    python indexer/build.py --source qdrant --dry-run --limit 200

Nothing here loads a model. Each point carries its text and the name of a model
that lives inside the Qdrant cluster, and the cluster returns the vectors. That
is the entire difference between this and the old indexer, and it is what lets
the repository drop torch, transformers, sentence-transformers, the Dockerfile
and the machine that used to run them.

Runs are resumable. A batch that has been acknowledged is recorded, so a run
interrupted at hour four picks up where it stopped instead of paying to embed
everything again.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_rest import call, scroll, tokens_by_model  # noqa: E402
from textifier import textify  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
STATE_DIR = os.path.join(HERE, ".state")

DENSE_MODEL = os.environ.get("QDRANT_DENSE_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
SPARSE_MODEL = os.environ.get("QDRANT_SPARSE_MODEL", "Qdrant/bm25")

# Cut every document to this many characters before embedding. 0 disables it.
#
# This exists because Qdrant Cloud Inference truncates at 512 tokens for every model,
# and all-MiniLM-L6-v2 has a published max_seq_length of 256 - it was trained at
# that length, and positions past it are ones it barely saw. Feeding it the full
# 512 costs real recall on code: bench/truncation.py measures 0.863 docstring
# recall@10 on full text against 0.937 when the input is cut to fit. 700
# characters is roughly 256 tokens of Rust.
#
# Models whose own limit is 512, mxbai-embed-large-v1 among them, do not need
# this and should run with it set to 0.
TEXT_CHAR_BUDGET = int(os.environ.get("INDEX_CHAR_BUDGET", "0"))

# Dimensions are a property of the model, so they are looked up rather than
# configured. A mismatch here is only discovered as "expected dim X, got Y" on
# the first upsert, tens of thousands of points into a run.
DENSE_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
    "qdrant/clip-vit-b-32-text": 512,
}

CODE_COLLECTION = os.environ.get("QDRANT_CODE_COLLECTION", "code-snippets-cloud")
NLU_COLLECTION = os.environ.get("QDRANT_NLU_COLLECTION", "code-signatures-cloud")
FILE_COLLECTION = os.environ.get("QDRANT_FILE_COLLECTION", "code-files-cloud")

# Old collections, read from when --source qdrant. Never written to.
OLD_CODE = os.environ.get("OLD_CODE_COLLECTION", "code-snippets-unixcoder")
OLD_NLU = os.environ.get("OLD_NLU_COLLECTION", "code-signatures")
OLD_FILES = os.environ.get("OLD_FILE_COLLECTION", "code-files")

BATCH = int(os.environ.get("INDEX_BATCH", "16"))
WORKERS = int(os.environ.get("INDEX_WORKERS", "6"))


# --------------------------------------------------------------------------- #
# sources


def snippets_from_qdrant():
    for payload in scroll(OLD_CODE):
        text = payload.get("code_snippet")
        if text:
            yield text, payload


def snippets_from_files():
    path = os.path.join(DATA_DIR, "qdrant_snippets.jsonl")
    # The old indexer took the first of these that was present, so the same
    # order is kept here - changing it would silently re-embed the corpus
    # against different text and make the two demos incomparable.
    keys = ("code_snippet", "body", "signature", "name")
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            body = next((row[k] for k in keys if row.get(k)), None)
            if not body:
                continue
            docstring = row.get("docstring") or ""
            yield f"{docstring} {body}".strip(), row


def signatures_from_qdrant():
    for payload in scroll(OLD_NLU):
        yield textify(payload), payload


def signatures_from_files():
    path = os.path.join(DATA_DIR, "structures.json")
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                row = json.loads(line)
                yield textify(row), row


def files_from_qdrant():
    for payload in scroll(OLD_FILES):
        yield None, payload


def files_from_files():
    path = os.path.join(DATA_DIR, "rs_files.json")
    with open(path, encoding="utf-8") as fp:
        for row in json.load(fp):
            yield None, row


TARGETS = {
    "snippets": {
        "collection": CODE_COLLECTION,
        "vectors": True,
        "qdrant": snippets_from_qdrant,
        "files": snippets_from_files,
        "payload_index": [],
    },
    "signatures": {
        "collection": NLU_COLLECTION,
        "vectors": True,
        "qdrant": signatures_from_qdrant,
        "files": signatures_from_files,
        "payload_index": [],
    },
    "files": {
        "collection": FILE_COLLECTION,
        "vectors": False,
        "qdrant": files_from_qdrant,
        "files": files_from_files,
        # /api/file is a filtered scroll on `path` and nothing else reads this
        # collection. Clusters with strict mode on refuse to filter an unindexed
        # field, so without this the file viewer fails with a 500 on every
        # result anyone clicks, while search itself keeps working - which makes
        # it look like a frontend problem.
        "payload_index": [("path", "keyword")],
    },
}


# --------------------------------------------------------------------------- #
# collection setup


def create(collection, with_vectors):
    if with_vectors:
        size = DENSE_DIMS.get(DENSE_MODEL)
        if size is None:
            raise SystemExit(
                f"Unknown dimensions for {DENSE_MODEL}. Add it to DENSE_DIMS; the "
                "Inference tab of the cluster page in the Cloud Console lists them."
            )
        body = {
            "vectors": {"dense": {"size": size, "distance": "Cosine", "on_disk": True}},
            # BM25 needs IDF applied at query time, computed by the engine from
            # collection statistics. That is also why the sparse leg costs no
            # inference tokens.
            "sparse_vectors": {"sparse": {"modifier": "idf"}},
            "quantization_config": {
                "scalar": {"type": "int8", "always_ram": True, "quantile": 0.99}
            },
        }
    else:
        # No vectors at all. This collection is the demo's file store, which is
        # here so that Qdrant is the only thing holding state.
        body = {"vectors": {}}

    call("DELETE", f"/collections/{collection}")
    call("PUT", f"/collections/{collection}", body)


def make_point(idx, text, payload, with_vectors):
    if not with_vectors:
        return {"id": idx, "vector": {}, "payload": payload}
    # Both legs read the same text. If the dense model is not being shown the
    # tail of a document, the lexical leg should not be either, or the two are
    # ranking different corpora.
    if TEXT_CHAR_BUDGET:
        text = text[:TEXT_CHAR_BUDGET]
    return {
        "id": idx,
        "vector": {
            "dense": {"text": text, "model": DENSE_MODEL},
            "sparse": {"text": text, "model": SPARSE_MODEL},
        },
        "payload": payload,
    }


# --------------------------------------------------------------------------- #
# run


def state_path(collection):
    return os.path.join(STATE_DIR, f"{collection}.json")


def load_state(collection, fresh):
    if fresh:
        return set()
    try:
        with open(state_path(collection), encoding="utf-8") as fp:
            return set(json.load(fp)["done"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return set()


def save_state(collection, done):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(state_path(collection), "w", encoding="utf-8") as fp:
        json.dump({"done": sorted(done)}, fp)


def batches(rows, size):
    batch = []
    for item in rows:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def run(name, source, args):
    target = TARGETS[name]
    collection = target["collection"]
    with_vectors = target["vectors"]

    print(f"\n=== {name} -> {collection} (source: {source})")
    rows = target[source]()
    if args.limit:
        rows = (r for i, r in enumerate(rows) if i < args.limit)

    done = load_state(collection, fresh=args.fresh)
    if args.fresh or not done:
        if args.dry_run:
            print(f"  would recreate {collection}")
        else:
            create(collection, with_vectors)
            print(f"  created {collection}")
    elif done:
        print(f"  resuming, {len(done)} batches already uploaded")

    lock = threading.Lock()
    totals, counts = {}, {"points": 0, "batches": 0, "skipped": 0}
    started = time.perf_counter()

    def upload(offset, batch):
        points = [
            make_point(offset + i, text, payload, with_vectors)
            for i, (text, payload) in enumerate(batch)
        ]
        _res, usage = call("PUT", f"/collections/{collection}/points?wait=true", {"points": points})
        with lock:
            done.add(offset)
            counts["points"] += len(points)
            counts["batches"] += 1
            for model, n in tokens_by_model(usage).items():
                totals[model] = totals.get(model, 0) + n
            if counts["batches"] % 50 == 0:
                elapsed = time.perf_counter() - started
                rate = counts["points"] / elapsed if elapsed else 0
                print(f"  {counts['points']:>7} points  {rate:6.1f}/s", flush=True)
                save_state(collection, done)

    # Resume works by batch offset, which is only meaningful because both
    # sources yield in a stable order: scroll returns points by id, and the
    # files are read start to end. Change either and a resumed run would write
    # different documents to ids it thinks are already done.
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = []
            for offset, batch in ((i * BATCH, b) for i, b in enumerate(batches(rows, BATCH))):
                if offset in done:
                    counts["skipped"] += 1
                    continue
                if args.dry_run:
                    counts["points"] += len(batch)
                    counts["batches"] += 1
                    continue
                futures.append(pool.submit(upload, offset, batch))
                # Bound the queue so a 123k-point run does not materialise every
                # batch in memory before the first one finishes.
                if len(futures) >= WORKERS * 4:
                    for future in as_completed(futures):
                        future.result()
                    futures = []
            for future in as_completed(futures):
                future.result()
    finally:
        # Save on the way out however the run ended. Losing the last few
        # hundred acknowledged batches to an exception means paying to embed
        # them again, which is the one cost this script can actually waste.
        if not args.dry_run:
            save_state(collection, done)

    if not args.dry_run:
        for field, schema in target["payload_index"]:
            call(
                "PUT",
                f"/collections/{collection}/index?wait=true",
                {"field_name": field, "field_schema": schema},
            )
            print(f"  indexed payload field `{field}` as {schema}")

    elapsed = time.perf_counter() - started
    print(
        f"  {counts['points']} points in {elapsed:.0f}s"
        + (f", {counts['skipped']} batches skipped" if counts["skipped"] else "")
    )
    if totals:
        print(f"  inference tokens: {json.dumps(totals)}")
    return {"collection": collection, "points": counts["points"], "seconds": round(elapsed, 1),
            "tokens": totals}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("qdrant", "files"), default="qdrant",
                    help="read the corpus from the old collections, or from data/*.json")
    ap.add_argument("--only", choices=tuple(TARGETS), action="append",
                    help="build one collection; repeatable")
    ap.add_argument("--limit", type=int, help="stop after N records, for a smoke run")
    ap.add_argument("--fresh", action="store_true",
                    help="recreate the collection and ignore any saved progress")
    ap.add_argument("--dry-run", action="store_true",
                    help="count the work without writing anything")
    args = ap.parse_args()

    names = args.only or list(TARGETS)
    print(f"dense: {DENSE_MODEL}   sparse: {SPARSE_MODEL}   batch: {BATCH} x {WORKERS} workers")
    print(f"text cut to {TEXT_CHAR_BUDGET} chars" if TEXT_CHAR_BUDGET else "text not cut")

    summary = [run(name, args.source, args) for name in names]

    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
