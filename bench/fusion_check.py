"""Is the sparse leg helping or hurting on the deployed corpus?

    python bench/fusion_check.py

The head-to-head against the old demo came out split: the rebuild wins the
docstring set by seven points of recall@5 and loses the paraphrase set. The
bake-off, on a 5,000-document pool, already hinted at why - mxbai on its own
scored 0.381 paraphrase recall@10 against 0.336 for mxbai fused with BM25. When
a query shares no identifiers with its answer, the lexical leg is not neutral,
it actively pushes wrong documents up.

This checks that on the real 123k corpus rather than the sample, by querying the
deployed signature collection three ways. It only reads, and it writes nothing.

The signature collection is the one under test because it is the collection that
produces results. The snippet collection only contributes highlight ranges.
"""

import json
import os
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))

from common import call, percentile  # noqa: E402

COLLECTION = os.environ.get("QDRANT_NLU_COLLECTION", "code-signatures-cloud")
DENSE_MODEL = os.environ.get("QDRANT_DENSE_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
SPARSE_MODEL = os.environ.get("QDRANT_SPARSE_MODEL", "Qdrant/bm25")

# The old demo's signature search, reproduced exactly: MiniLM over the
# collection the old deployment still reads. Included so the comparison is
# between two rankings measured the same way on the same day, rather than
# against a figure published earlier under unknown conditions.
OLD_COLLECTION = os.environ.get("OLD_NLU_COLLECTION", "code-signatures")
OLD_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EVAL_DIR = os.environ.get("EVAL_DIR", r"C:\Users\Home Laptop\code-search-eval")
RESULTS = os.path.join(os.path.dirname(__file__), "results")

# The demo shows five results, so five is what gets scored. Ranking quality
# below the fold is not what a visitor experiences.
LIMIT = 5
PREFETCH = 100
KS = (1, 3, 5)
WORKERS = 4


def load(name):
    with open(os.path.join(EVAL_DIR, name), encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def dense_leg(text):
    return {"query": {"text": text, "model": DENSE_MODEL}, "using": "dense", "limit": PREFETCH}


def sparse_leg(text):
    return {"query": {"text": text, "model": SPARSE_MODEL}, "using": "sparse", "limit": PREFETCH}


def body(config, text):
    payload = ["name", "context"]
    if config == "dense only":
        leg = dense_leg(text)
        return {"query": leg["query"], "using": "dense", "limit": LIMIT, "with_payload": payload}
    if config == "sparse only":
        leg = sparse_leg(text)
        return {"query": leg["query"], "using": "sparse", "limit": LIMIT, "with_payload": payload}
    if config == "hybrid RRF":
        return {"prefetch": [dense_leg(text), sparse_leg(text)],
                "query": {"fusion": "rrf"}, "limit": LIMIT, "with_payload": payload}
    # Dense decides the ranking; the sparse leg only widens the candidate pool
    # it reranks. A lexical hit can still surface a document, but it cannot
    # push one to the top on term overlap alone.
    if config == "sparse recall, dense rank":
        return {"prefetch": [dense_leg(text), sparse_leg(text)],
                "query": {"text": text, "model": DENSE_MODEL}, "using": "dense",
                "limit": LIMIT, "with_payload": payload}
    if config == "old demo (minilm)":
        return {"query": {"text": text, "model": OLD_MODEL},
                "limit": LIMIT, "with_payload": payload}
    raise ValueError(config)


def collection_for(config):
    return OLD_COLLECTION if config == "old demo (minilm)" else COLLECTION


CONFIGS = ["dense only", "sparse only", "hybrid RRF", "sparse recall, dense rank",
           "old demo (minilm)"]


def rank_of(points, expected_file, expected_name):
    for rank, point in enumerate(points, start=1):
        context = point["payload"].get("context") or {}
        if context.get("file_path") != expected_file:
            continue
        name = point["payload"].get("name")
        if not name or not expected_name or name == expected_name:
            return rank
    return None


def evaluate(config, queries):
    ranks, times = [None] * len(queries), []

    collection = collection_for(config)

    def one(i):
        result, ms, _u = call("POST", f"/collections/{collection}/points/query",
                              body(config, queries[i]["query"]))
        times.append(ms)
        ranks[i] = rank_of(result["points"], queries[i]["file_path"], queries[i].get("name"))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(one, range(len(queries))))

    n = len(queries)
    return {
        **{f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 4) for k in KS},
        "mrr@5": round(sum(1.0 / r for r in ranks if r and r <= LIMIT) / n, 4),
        "p50_ms": percentile(times, 50),
        # Kept so two configurations can be compared query by query. A
        # difference in the averages says nothing about whether the two rankings
        # actually differ on this many queries.
        "ranks": ranks,
    }


def main():
    sets = {"docstring": load("queries.jsonl"), "paraphrase": load("queries_paraphrase.jsonl")}
    print(f"{COLLECTION}, top {LIMIT}, "
          + ", ".join(f"{len(v)} {k}" for k, v in sets.items()))

    report = {}
    for config in CONFIGS:
        report[config] = {}
        for set_name, queries in sets.items():
            report[config][set_name] = evaluate(config, queries)
        d, p = report[config]["docstring"], report[config]["paraphrase"]
        print(f"  {config:28} doc R@5 {d['recall@5']:.3f} MRR {d['mrr@5']:.3f}  |  "
              f"par R@5 {p['recall@5']:.3f} MRR {p['mrr@5']:.3f}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "fusion_check.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    header = (f"{'configuration':28} {'docR@1':>7} {'docR@5':>7} {'docMRR':>7} "
              f"{'parR@1':>7} {'parR@5':>7} {'parMRR':>7} {'p50':>6}")
    print("\n" + header)
    print("-" * len(header))
    for config in CONFIGS:
        d, p = report[config]["docstring"], report[config]["paraphrase"]
        print(f"{config:28} {d['recall@1']:>7.3f} {d['recall@5']:>7.3f} {d['mrr@5']:>7.3f} "
              f"{p['recall@1']:>7.3f} {p['recall@5']:>7.3f} {p['mrr@5']:>7.3f} "
              f"{d['p50_ms']:>6.0f}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
