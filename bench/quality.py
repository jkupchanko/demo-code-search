"""Score a deployed demo end to end, on the same queries as the old one.

    python bench/quality.py --target https://demo-code-search-cloud.vercel.app
    python bench/quality.py --target <new> --compare https://code-search.qdrant.tech

This is the number that decides whether the rebuild is allowed to ship: the old
demo scored 0.907 file-level recall@10 over the full 123k corpus, and a
consolidation that quietly costs a tenth of the answers is not a consolidation
worth having.

Both endpoints return five results per search, so recall@10 cannot exceed
recall@5 for either of them. That is a property of the demo, not of the models,
and it is left alone here so the two runs stay comparable.

Two query sets, and they disagree:

  queries.jsonl            docstrings taken from the code they describe, so they
                           share identifiers with the answer
  queries_paraphrase.jsonl the same intent rewritten to avoid those identifiers,
                           which is closer to what a person types
"""

import argparse
import json
import os
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))

from common import Pool, percentile, target_headers  # noqa: E402

EVAL_DIR = os.environ.get("EVAL_DIR", r"C:\Users\Home Laptop\code-search-eval")
RESULTS = os.path.join(os.path.dirname(__file__), "results")
KS = (1, 5, 10)


def load(name):
    with open(os.path.join(EVAL_DIR, name), encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def search(pool, prefix, query):
    path = f"{prefix}/api/search?query={urllib.parse.quote(query)}"
    status, raw, ms = pool.get(path, target_headers())
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {raw[:200].decode(errors='replace')}")
    return json.loads(raw).get("result", []), ms


def ranks_for(hits, expected_file, expected_name):
    """Rank of the right answer, strictly and by file alone.

    Strict means the hit names the same function in the same file. File-only
    means the search landed in the right file, which is what the old demo's
    published figure measured, so both are kept rather than picking one.
    """
    strict = loose = None
    for rank, hit in enumerate(hits, start=1):
        context = hit.get("context") or {}
        path = context.get("file_path") or hit.get("file")
        if path != expected_file:
            continue
        if loose is None:
            loose = rank
        name = hit.get("name")
        if strict is None and (not name or not expected_name or name == expected_name):
            strict = rank
    return strict, loose


def score(ranks, n):
    out = {f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 4) for k in KS}
    out["mrr@10"] = round(sum(1.0 / r for r in ranks if r and r <= 10) / n, 4)
    return out


def evaluate(base, queries, workers):
    strict, loose, times = [None] * len(queries), [None] * len(queries), []
    errors = []
    url = base if "://" in base else f"https://{base}"
    # One Pool shared by every worker; it hands out a connection per thread, so
    # `workers` connections are opened once instead of one per request.
    http = Pool(url, timeout=90)
    prefix = urllib.parse.urlparse(url).path.rstrip("/")

    def one(i):
        q = queries[i]
        try:
            hits, ms = search(http, prefix, q["query"])
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{i}: {exc}")
            return
        times.append(ms)
        strict[i], loose[i] = ranks_for(hits, q["file_path"], q.get("name"))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(one, range(len(queries))))

    n = len(queries)
    return {
        "queries": n,
        "errors": len(errors),
        "error_sample": errors[:3],
        "strict": score(strict, n),
        "file_only": score(loose, n),
        # Measured with `workers` requests in flight, so this is throughput
        # latency, not the single-user figure. bench/latency.py measures that.
        "concurrent_p50_ms": percentile(times, 50),
        "concurrent_p95_ms": percentile(times, 95),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--compare", help="another deployment to score on the same queries")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="use only the first N queries of each set")
    args = ap.parse_args()

    sets = {"docstring": load("queries.jsonl"), "paraphrase": load("queries_paraphrase.jsonl")}
    if args.limit:
        sets = {k: v[: args.limit] for k, v in sets.items()}

    report = {}
    for label, base in [("rebuild", args.target), ("comparison", args.compare)]:
        if not base:
            continue
        print(f"\n{label}: {base}")
        report[label] = {"target": base}
        for set_name, queries in sets.items():
            print(f"  {set_name} ({len(queries)} queries) ...", flush=True)
            report[label][set_name] = evaluate(base, queries, args.workers)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "quality.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    hdr = f"{'target':12} {'set':11} {'R@1':>7} {'R@5':>7} {'fileR@5':>8} {'MRR':>7} {'err':>4}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, row in report.items():
        for set_name in sets:
            r = row[set_name]
            print(f"{label:12} {set_name:11} {r['strict']['recall@1']:>7.3f} "
                  f"{r['strict']['recall@5']:>7.3f} {r['file_only']['recall@5']:>8.3f} "
                  f"{r['strict']['mrr@10']:>7.3f} {r['errors']:>4}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
