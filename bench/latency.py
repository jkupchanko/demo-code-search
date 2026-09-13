"""Latency of a deployed search endpoint, and of the Qdrant leg underneath it.

    python bench/latency.py --target https://demo-code-search-cloud.vercel.app
    python bench/latency.py --target http://127.0.0.1:3000 --n 100
    python bench/latency.py --target <url> --compare https://code-search.qdrant.tech

Reports three numbers that are usually conflated:

  end-to-end   what a viewer waits for, measured from this machine
  server       what the function reports it spent talking to Qdrant
  overhead     the difference, which is the function, Vercel's edge, and the
               network between here and there

Splitting them matters because the rebuild moved the embedding from the backend
process into the cluster. If only the total is reported, a slower network reads
as a slower search and an improvement in the search reads as noise.

The first request is reported on its own rather than averaged in. It pays the
connection setup and any serverless cold start, which is a different event from
the hundredth request, and burying it in a p50 hides the one number people ask
about when they hear the backend is serverless.
"""

import argparse
import json
import os
import statistics
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(__file__))

from common import Pool, percentile, target_headers  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")

# Queries the demo itself suggests, plus a few that are deliberately awkward:
# an exact identifier, a misspelling, and questions with no lexical overlap with
# any answer. A latency table built only from the happy path is a table of
# cache hits.
QUERIES = [
    "cardinality of should request",
    "how to calculate the size of a quantized vector",
    "flush the write ahead log to disk",
    "estimate_cardinality",
    "recomend points based on positive and negative examples",
    "what happens when a shard is transferred to another node",
    "convert grpc filter into internal representation",
    "throttle the optimizer so it does not eat the machine",
    "read a segment while it is being written",
    "where are payload indexes persisted",
]


def normalise(base):
    """Split a base URL into a Pool and the path prefix to prepend."""
    url = base if "://" in base else f"https://{base}"
    parsed = urllib.parse.urlparse(url)
    return Pool(url, timeout=90), parsed.path.rstrip("/")


def search(pool, prefix, query):
    """One search against the deployment. Returns (body, status, elapsed_ms).

    The connection is kept alive across calls, so these samples measure the
    search rather than a TLS handshake - which on a cross-region link is larger
    than most of what is being compared.
    """
    path = f"{prefix}/api/search?query={urllib.parse.quote(query)}"
    status, raw, ms = pool.get(path, target_headers())
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        body = {"detail": raw[:200].decode(errors="replace")}
    return body, status, ms


def measure(base, n, label):
    print(f"\n{label}  {base}")
    pool, prefix = normalise(base)

    first, status, first_ms = search(pool, prefix, "cold start probe, not counted below")
    if status != 200:
        print(f"  FAILED {status}: {json.dumps(first)[:200]}")
        return None
    print(f"  first request (connection setup and any cold start): {first_ms:.0f} ms")

    end_to_end, server, errors, tokens = [], [], 0, 0
    for i in range(n):
        # Vary the query so the run measures search rather than a warm cache.
        # Repeating one string would report how fast Qdrant returns something it
        # has already computed, which is not what anyone is asking about.
        suffix = i // len(QUERIES)
        query = f"{QUERIES[i % len(QUERIES)]} {suffix or ''}".strip()
        body, status, ms = search(pool, prefix, query)
        if status != 200:
            errors += 1
            continue
        end_to_end.append(ms)
        if isinstance(body.get("latency_ms"), (int, float)):
            server.append(float(body["latency_ms"]))
        tokens += body.get("inference_tokens") or 0

    if not end_to_end:
        print(f"  every request failed ({errors})")
        return None

    row = {
        "target": base,
        "requests": len(end_to_end),
        "errors": errors,
        "first_request_ms": round(first_ms),
        "end_to_end": {
            "p50": percentile(end_to_end, 50),
            "p95": percentile(end_to_end, 95),
            "mean": round(statistics.fmean(end_to_end), 1),
        },
        "inference_tokens_total": tokens,
    }
    if server:
        row["server"] = {"p50": percentile(server, 50), "p95": percentile(server, 95)}
        row["overhead_p50"] = round(row["end_to_end"]["p50"] - row["server"]["p50"], 1)

    print(f"  end-to-end   p50 {row['end_to_end']['p50']:>7.1f} ms   "
          f"p95 {row['end_to_end']['p95']:>7.1f} ms")
    if server:
        print(f"  server       p50 {row['server']['p50']:>7.1f} ms   "
              f"p95 {row['server']['p95']:>7.1f} ms")
        print(f"  overhead     p50 {row['overhead_p50']:>7.1f} ms")
    else:
        print("  server       not reported by this endpoint")
    if tokens:
        print(f"  inference tokens over {len(end_to_end)} searches: {tokens}")
    if errors:
        print(f"  {errors} request(s) failed")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="base URL of the rebuilt demo")
    ap.add_argument("--compare", help="base URL of another deployment, e.g. the old demo")
    ap.add_argument("--n", type=int, default=50, help="requests per target")
    args = ap.parse_args()

    rows = [r for r in [
        measure(args.target, args.n, "rebuild"),
        measure(args.compare, args.n, "comparison") if args.compare else None,
    ] if r]

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "latency.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
