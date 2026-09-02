"""Is a difference between two configurations real, or is it 113 queries?

    python bench/significance.py "hybrid RRF" "old demo (minilm)"
    python bench/significance.py "dense only" "hybrid RRF" --set docstring

Reads the per-query ranks bench/fusion_check.py saved and compares two
configurations query by query, which the averages cannot do. A gap of 0.02 MRR
across 113 queries can be one ranking genuinely beating another, or it can be
four queries landing differently.

Two things are reported. A paired bootstrap over the query set gives a
confidence interval on the difference: if it straddles zero, the two are not
distinguishable on this many queries. An exact sign test over the queries where
the two disagree gives the probability of seeing a split that lopsided by
chance.

Stdlib only, and it reads a file rather than the network, so it is instant and
repeatable.
"""

import argparse
import json
import math
import os
import random
import sys

RESULTS = os.path.join(os.path.dirname(__file__), "results")
LIMIT = 5
BOOTSTRAP = 20000
SEED = 20260901


def reciprocal(rank):
    return 1.0 / rank if rank and rank <= LIMIT else 0.0


def mrr(ranks):
    return sum(reciprocal(r) for r in ranks) / len(ranks)


def bootstrap(a, b, rounds, rng):
    """Paired bootstrap over queries. Returns the 95% interval on mrr(a) - mrr(b)."""
    n = len(a)
    diffs = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(
            sum(reciprocal(a[i]) for i in idx) / n - sum(reciprocal(b[i]) for i in idx) / n
        )
    diffs.sort()
    return diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds)]


def sign_test(a, b):
    """Two-sided exact sign test over the queries where the two rankings differ."""
    wins = sum(1 for x, y in zip(a, b) if reciprocal(x) > reciprocal(y))
    losses = sum(1 for x, y in zip(a, b) if reciprocal(x) < reciprocal(y))
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return wins, losses, min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--set", dest="query_set", default=None,
                    help="docstring or paraphrase; both if omitted")
    args = ap.parse_args()

    path = os.path.join(RESULTS, "fusion_check.json")
    with open(path, encoding="utf-8") as fp:
        report = json.load(fp)

    for name in (args.first, args.second):
        if name not in report:
            sys.exit(f"{name!r} not in {path}. Available: {', '.join(report)}")

    sets = [args.query_set] if args.query_set else ["docstring", "paraphrase"]
    rng = random.Random(SEED)

    print(f"{args.first}  vs  {args.second}\n")
    header = f"{'set':12} {'MRR a':>7} {'MRR b':>7} {'diff':>8} {'95% interval':>22} {'sign p':>8}"
    print(header)
    print("-" * len(header))

    for name in sets:
        a = report[args.first][name]["ranks"]
        b = report[args.second][name]["ranks"]
        if len(a) != len(b):
            sys.exit(f"{name}: rank vectors differ in length, {len(a)} vs {len(b)}")
        lo, hi = bootstrap(a, b, BOOTSTRAP, rng)
        wins, losses, p = sign_test(a, b)
        verdict = "" if lo <= 0 <= hi else "  <- excludes zero"
        print(f"{name:12} {mrr(a):>7.3f} {mrr(b):>7.3f} {mrr(a) - mrr(b):>8.3f} "
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22} {p:>8.3f}{verdict}")
        print(f"{'':12} {wins} queries better, {losses} worse, "
              f"{len(a) - wins - losses} identical")

    print(f"\npaired bootstrap, {BOOTSTRAP} resamples, seed {SEED}; "
          "sign test is two-sided and exact")


if __name__ == "__main__":
    main()
