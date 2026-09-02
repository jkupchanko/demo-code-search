"""Pick the encoder for the rebuild, measured on Qdrant Cloud Inference itself.

The old demo embedded queries with UnixCoder in the backend process. UnixCoder
is not in the Qdrant Cloud Inference catalog and cannot be added, so consolidating onto
one vendor forces an encoder change. This decides which one, and it does it by
running the real production path - the cluster embeds, the cluster searches -
rather than by scoring vectors in a notebook.

Two query sets, because they disagree and the disagreement is the finding:

  queries.jsonl            docstrings lifted from the code they describe, so they
                           share tokens with the answer. Flatters lexical search.
  queries_paraphrase.jsonl the same intent rewritten to avoid the identifiers.
                           This is what a person actually types.

Writes one temporary collection and deletes it at the end. Touches nothing the
live demo reads.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from common import (  # noqa: E402
    DENSE,
    SPARSE,
    call,
    percentile,
    tokens_by_model,
)

COLLECTION = "bench-code-cloudinf"
EVAL_DIR = os.environ.get("EVAL_DIR", r"C:\Users\Home Laptop\code-search-eval")
RESULTS = os.path.join(os.path.dirname(__file__), "results")

PREFETCH = 100
LIMIT = 10
KS = (1, 5, 10)
BATCH = 16


def load(name):
    with open(os.path.join(EVAL_DIR, name), encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def build(corpus):
    """Create the bench collection and embed every document with all four models.

    One upsert per batch carries all four vectors, so a document is read once and
    the four models see byte-identical input. Any difference in the results below
    is the model, not the preprocessing.
    """
    call("DELETE", f"/collections/{COLLECTION}")
    call(
        "PUT",
        f"/collections/{COLLECTION}",
        {
            "vectors": {
                name: {"size": size, "distance": "Cosine", "on_disk": True}
                for name, (_model, size) in DENSE.items()
            },
            # BM25 needs IDF applied at query time; the engine computes it from
            # collection statistics, which is why it costs no inference tokens.
            "sparse_vectors": {
                "bm25": {"modifier": "idf"},
                "splade": {},
            },
        },
    )

    totals = {}
    embed_ms = []
    for start in range(0, len(corpus), BATCH):
        batch = corpus[start : start + BATCH]
        points = []
        for i, doc in enumerate(batch, start=start):
            text = doc["text"]
            vector = {
                name: {"text": text, "model": model} for name, (model, _s) in DENSE.items()
            }
            vector.update(
                {name: {"text": text, "model": model} for name, model in SPARSE.items()}
            )
            points.append(
                {
                    "id": i,
                    "vector": vector,
                    "payload": {"doc_id": doc["id"], "file_path": doc["file_path"]},
                }
            )
        _res, ms, usage = call("PUT", f"/collections/{COLLECTION}/points?wait=true", {"points": points})
        embed_ms.append(ms)
        for model, n in tokens_by_model(usage).items():
            totals[model] = totals.get(model, 0) + n
        done = min(start + BATCH, len(corpus))
        if done % 800 == 0 or done == len(corpus):
            print(f"  indexed {done}/{len(corpus)}", flush=True)

    return totals, embed_ms


def query_body(config, text):
    """Build the query for one configuration.

    Hybrid runs as a server-side prefetch per leg plus a fusion step, which is
    one round trip - the same shape the deployed API will use.
    """
    dense, sparse, fusion = config
    legs = []
    if dense:
        model, _size = DENSE[dense]
        legs.append({"query": {"text": text, "model": model}, "using": dense, "limit": PREFETCH})
    if sparse:
        legs.append(
            {"query": {"text": text, "model": SPARSE[sparse]}, "using": sparse, "limit": PREFETCH}
        )

    if len(legs) == 1:
        leg = legs[0]
        return {"query": leg["query"], "using": leg["using"], "limit": LIMIT,
                "with_payload": ["doc_id"]}
    return {"prefetch": legs, "query": {"fusion": fusion}, "limit": LIMIT,
            "with_payload": ["doc_id"]}


def rank_of(points, answer_id):
    for rank, point in enumerate(points, start=1):
        if point["payload"]["doc_id"] == answer_id:
            return rank
    return None


def metrics(ranks):
    n = len(ranks)
    out = {f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 4) for k in KS}
    out["mrr@10"] = round(sum(1.0 / r for r in ranks if r and r <= 10) / n, 4)
    return out


def evaluate(config, queries):
    ranks, latencies, tokens = [], [], 0
    for q in queries:
        result, ms, usage = call(
            "POST", f"/collections/{COLLECTION}/points/query", query_body(config, q["query"])
        )
        ranks.append(rank_of(result["points"], q["answer_id"]))
        latencies.append(ms)
        tokens += sum(tokens_by_model(usage).values())
    return {
        **metrics(ranks),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "tokens": tokens,
    }


# (dense leg, sparse leg, fusion). RRF ranks by position, so a document has to
# place well in one leg to survive; DBSF normalises the two score distributions
# and adds them, which lets a strong dense score outvote a weak lexical one.
# Both are tested because the paraphrase set is exactly where that difference
# should show: those queries share no identifiers with their answers, so the
# lexical leg is contributing noise and RRF still gives it half the vote.
CONFIGS = [
    ("minilm", None, None),
    ("mxbai", None, None),
    (None, "bm25", None),
    (None, "splade", None),
    ("minilm", "bm25", "rrf"),
    ("minilm", "splade", "rrf"),
    ("mxbai", "bm25", "rrf"),
    ("mxbai", "splade", "rrf"),
    ("minilm", "bm25", "dbsf"),
    ("mxbai", "bm25", "dbsf"),
]


def label(config):
    dense, sparse, fusion = config
    if dense and sparse:
        return f"{dense} + {sparse} ({fusion.upper()})"
    return f"{dense or sparse} only"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-index", action="store_true",
                    help="reuse an existing bench collection")
    ap.add_argument("--keep", action="store_true",
                    help="leave the bench collection in place afterwards")
    ap.add_argument("--resume", action="store_true",
                    help="keep configurations already scored in results/bakeoff.json")
    args = ap.parse_args()

    corpus = load("corpus.jsonl")
    sets = {
        "docstring": load("queries.jsonl"),
        "paraphrase": load("queries_paraphrase.jsonl"),
    }
    print(f"{len(corpus)} documents, " + ", ".join(f"{len(v)} {k}" for k, v in sets.items()))

    index_tokens, embed_ms = {}, []
    if not args.skip_index:
        print(f"indexing into {COLLECTION} with all four models ...", flush=True)
        index_tokens, embed_ms = build(corpus)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "bakeoff.json")

    # Rows already scored are kept. Eight configurations over two query sets is
    # a few thousand requests, and losing the completed half to a reset
    # connection in the last one is a bad way to spend an hour.
    report = {"corpus_size": len(corpus), "prefetch_per_leg": PREFETCH, "rows": []}
    if args.resume and os.path.exists(out):
        with open(out, encoding="utf-8") as fp:
            report = json.load(fp)
        print(f"resuming, {len(report['rows'])} configurations already scored")
    if index_tokens:
        report["index_tokens"] = index_tokens
        report["index_batch_p50_ms"] = percentile(embed_ms, 50)

    scored = {row["config"] for row in report["rows"]}
    for config in CONFIGS:
        name = label(config)
        if name in scored:
            continue
        row = {"config": name}
        for set_name, queries in sets.items():
            print(f"  {name:26} {set_name} ...", flush=True)
            row[set_name] = evaluate(config, queries)
        report["rows"].append(row)
        with open(out, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)

    rows = report["rows"]
    index_tokens = report.get("index_tokens", index_tokens)

    hdr = (f"{'configuration':26} {'docR@1':>7} {'docR@10':>8} {'docMRR':>7} "
           f"{'parR@1':>7} {'parR@10':>8} {'parMRR':>7} {'p50':>7} {'p95':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        d, p = r["docstring"], r["paraphrase"]
        print(f"{r['config']:26} {d['recall@1']:>7.3f} {d['recall@10']:>8.3f} {d['mrr@10']:>7.3f} "
              f"{p['recall@1']:>7.3f} {p['recall@10']:>8.3f} {p['mrr@10']:>7.3f} "
              f"{d['p50_ms']:>7.1f} {d['p95_ms']:>7.1f}")
    print(f"\nindex tokens: {json.dumps(index_tokens)}")
    print(f"written to {out}")

    if not args.keep:
        call("DELETE", f"/collections/{COLLECTION}")
        print(f"dropped {COLLECTION}")


if __name__ == "__main__":
    main()
