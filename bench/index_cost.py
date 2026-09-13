"""What a full indexing run costs, measured rather than guessed.

    python bench/index_cost.py                 # sample of 300 docs
    python bench/index_cost.py --sample 1000 --corpus-size 140444

Qdrant Cloud Inference bills per million tokens and reports the tokens it used on every
response. So the cost of embedding 140k chunks is a sample of a few hundred,
multiplied. Sampling is uniform across the corpus rather than taking the first N,
because the first N are alphabetically clustered and code snippet lengths are not
evenly distributed across a repository.

Writes one small collection and deletes it. Prices are not hardcoded: the
catalog changes and a stale number in a repository is worse than no number, so
this reports tokens and leaves the multiplication to whoever reads the Inference
tab of the cluster page.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from common import DENSE, SPARSE, call, percentile, tokens_by_model  # noqa: E402

COLLECTION = "bench-index-cost"
EVAL_DIR = os.environ.get("EVAL_DIR", r"C:\Users\Home Laptop\code-search-eval")
RESULTS = os.path.join(os.path.dirname(__file__), "results")

# The two collections a real run builds, and how many points each holds today.
CORPUS = {"code-snippets": 123257, "code-signatures": 17187}
BATCH = 16


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--corpus-size", type=int, default=sum(CORPUS.values()),
                    help="points a full run would embed")
    args = ap.parse_args()

    with open(os.path.join(EVAL_DIR, "corpus.jsonl"), encoding="utf-8") as fp:
        corpus = [json.loads(line) for line in fp if line.strip()]

    step = max(1, len(corpus) // args.sample)
    sample = corpus[::step][: args.sample]
    print(f"sampling {len(sample)} of {len(corpus)} documents, every {step}th")

    call("DELETE", f"/collections/{COLLECTION}")
    call("PUT", f"/collections/{COLLECTION}", {
        "vectors": {n: {"size": s, "distance": "Cosine"} for n, (_m, s) in DENSE.items()},
        "sparse_vectors": {"bm25": {"modifier": "idf"}, "splade": {}},
    })

    totals, batch_ms = {}, []
    try:
        for start in range(0, len(sample), BATCH):
            points = []
            for i, doc in enumerate(sample[start : start + BATCH], start=start):
                vector = {n: {"text": doc["text"], "model": m} for n, (m, _s) in DENSE.items()}
                vector.update({n: {"text": doc["text"], "model": m} for n, m in SPARSE.items()})
                points.append({"id": i, "vector": vector, "payload": {}})
            _res, ms, usage = call("PUT", f"/collections/{COLLECTION}/points?wait=true",
                                   {"points": points})
            batch_ms.append(ms)
            for model, n in tokens_by_model(usage).items():
                totals[model] = totals.get(model, 0) + n
    finally:
        call("DELETE", f"/collections/{COLLECTION}")

    n = len(sample)
    report = {
        "sampled": n,
        "corpus_size": args.corpus_size,
        "batch_p50_ms": percentile(batch_ms, 50),
        "models": {
            model: {
                "tokens_per_doc": round(tokens / n, 1),
                "projected_tokens": round(tokens / n * args.corpus_size),
                "projected_millions": round(tokens / n * args.corpus_size / 1e6, 2),
            }
            for model, tokens in sorted(totals.items())
        },
    }
    # Any model that was sent input but reported no tokens billed nothing. BM25
    # is the one that matters: the engine computes it from collection
    # statistics, so the sparse half of a hybrid index is free however large the
    # corpus gets. Compared on full model ids, lowercased, because that is what
    # the usage block echoes back.
    sent = {model.lower() for model, _dim in DENSE.values()} | {
        model.lower() for model in SPARSE.values()
    }
    report["free"] = sorted(sent - {k.lower() for k in totals})

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "index_cost.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print(f"\n{'model':40} {'tok/doc':>8} {'for ' + str(args.corpus_size):>14}")
    print("-" * 66)
    for model, row in report["models"].items():
        print(f"{model:40} {row['tokens_per_doc']:>8.1f} {row['projected_millions']:>11.2f} M")
    print(f"\nbilled nothing: {', '.join(report['free']) or 'none'}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
