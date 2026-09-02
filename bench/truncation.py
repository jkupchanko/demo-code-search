"""Does Qdrant Cloud Inference's 512-token window hurt all-MiniLM-L6-v2 on code?

    python bench/truncation.py

Background. The bake-off scored the in-cluster MiniLM at 0.863 docstring
recall@10. An earlier offline run of the same model, over the same 5,000
documents and the same 300 queries, scored 0.933. Same weights, same corpus,
seven points apart, which is too large to shrug at.

Probing the service with progressively longer input shows it truncates at
**512 tokens**. The published `sentence-transformers` configuration for this
model sets `max_seq_length` to **256**: it was trained at that length, and
positions past it are ones it barely saw. A third of this corpus is longer than
that, so the two runs were not feeding the model the same thing.

This turns the hypothesis into a measurement. It indexes the same corpus again
with every document cut down before it is sent, embeds it with the same
in-cluster MiniLM, and scores it against the same queries - on its own and in
the hybrid shape the demo ships. If the score climbs back toward the offline
figure, the window is the cause, and the fix for anyone using MiniLM through
Qdrant Cloud Inference is to cut their own input first.

Writes one temporary collection and drops it.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from common import DENSE, SPARSE, call, percentile, tokens_by_model  # noqa: E402

COLLECTION = "bench-code-trunc"
EVAL_DIR = os.environ.get("EVAL_DIR", r"C:\Users\Home Laptop\code-search-eval")
RESULTS = os.path.join(os.path.dirname(__file__), "results")

MODEL, DIMS = DENSE["minilm"]
TOKEN_BUDGET = 256
# The service reports tokens only after the fact, so the cut is made locally and
# verified against what comes back. A first attempt cut at 150 words assuming
# ~1.6 tokens per word, and the verification caught it: Rust tokenizes at closer
# to 3.4 tokens per whitespace-separated word once punctuation and split
# identifiers are counted, so those documents were still hitting the 512 cap.
# Characters are the steadier proxy, at roughly 2.9 per token on this corpus.
CHAR_BUDGET = 700
BATCH = 16
PREFETCH = 100
LIMIT = 10
KS = (1, 5, 10)


def load(name):
    with open(os.path.join(EVAL_DIR, name), encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def truncate(text):
    return text[:CHAR_BUDGET]


def query_body(leg, text):
    if leg == "dense":
        return {"query": {"text": text, "model": MODEL}, "using": "dense",
                "limit": LIMIT, "with_payload": ["doc_id"]}
    return {
        "prefetch": [
            {"query": {"text": text, "model": MODEL}, "using": "dense", "limit": PREFETCH},
            {"query": {"text": text, "model": SPARSE["bm25"]}, "using": "sparse",
             "limit": PREFETCH},
        ],
        "query": {"fusion": "rrf"},
        "limit": LIMIT,
        "with_payload": ["doc_id"],
    }


def main():
    corpus = load("corpus.jsonl")
    sets = {"docstring": load("queries.jsonl"), "paraphrase": load("queries_paraphrase.jsonl")}

    cut = sum(1 for doc in corpus if len(doc["text"]) > CHAR_BUDGET)
    print(f"{len(corpus)} documents, {cut} ({cut / len(corpus):.1%}) shortened to "
          f"{CHAR_BUDGET} characters")

    # A sparse leg as well, so the truncated dense model can be scored in the
    # hybrid shape the demo actually ships, not only on its own. BM25 reads the
    # same truncated text: whatever the dense model was not shown, the lexical
    # leg was not shown either, or the two are ranking different corpora.
    call("DELETE", f"/collections/{COLLECTION}")
    call("PUT", f"/collections/{COLLECTION}", {
        "vectors": {"dense": {"size": DIMS, "distance": "Cosine"}},
        "sparse_vectors": {"sparse": {"modifier": "idf"}},
    })

    max_tokens, total = 0, 0
    for start in range(0, len(corpus), BATCH):
        points = []
        for i, doc in enumerate(corpus[start : start + BATCH], start=start):
            text = truncate(doc["text"])
            points.append({
                "id": i,
                "vector": {
                    "dense": {"text": text, "model": MODEL},
                    "sparse": {"text": text, "model": SPARSE["bm25"]},
                },
                "payload": {"doc_id": doc["id"]},
            })
        _r, _ms, usage = call("PUT", f"/collections/{COLLECTION}/points?wait=true",
                              {"points": points})
        used = sum(tokens_by_model(usage).values())
        total += used
        max_tokens = max(max_tokens, used / max(len(points), 1))
        if (start + BATCH) % 1600 == 0:
            print(f"  indexed {min(start + BATCH, len(corpus))}/{len(corpus)}", flush=True)

    mean_tokens = total / len(corpus)
    under_budget = max_tokens <= TOKEN_BUDGET
    print(f"  mean {mean_tokens:.0f} tokens/doc, worst batch mean {max_tokens:.0f} "
          f"(budget {TOKEN_BUDGET})")
    if not under_budget:
        # Reported rather than silently accepted. The cut is per-document and the
        # check is per-batch, so a batch of unusually dense code can still average
        # over; the result below is still a cut-input measurement, just not a
        # strictly-under-256 one.
        print("  note: some batches averaged over the budget; lower CHAR_BUDGET to tighten")

    report = {
        "char_budget": CHAR_BUDGET,
        "documents_shortened": cut,
        "mean_tokens_per_doc": round(mean_tokens, 1),
        "worst_batch_mean_tokens": round(max_tokens, 1),
        "under_budget": under_budget,
        "sets": {},
    }

    for leg in ("dense", "hybrid"):
        for set_name, queries in sets.items():
            ranks, times = [], []
            for q in queries:
                result, ms, _u = call("POST", f"/collections/{COLLECTION}/points/query",
                                      query_body(leg, q["query"]))
                rank = next((r for r, point in enumerate(result["points"], 1)
                             if point["payload"]["doc_id"] == q["answer_id"]), None)
                ranks.append(rank)
                times.append(ms)
            n = len(queries)
            key = f"{leg}/{set_name}"
            report["sets"][key] = {
                **{f"recall@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 4)
                   for k in KS},
                "mrr@10": round(sum(1.0 / r for r in ranks if r and r <= 10) / n, 4),
                "p50_ms": percentile(times, 50),
            }
            print(f"  {key}: {json.dumps(report['sets'][key])}", flush=True)

    call("DELETE", f"/collections/{COLLECTION}")

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "truncation.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    # The full-text rows, so the comparison reads without a second file open.
    baseline = {}
    try:
        with open(os.path.join(RESULTS, "bakeoff.json"), encoding="utf-8") as fp:
            for row in json.load(fp)["rows"]:
                if row["config"] in ("minilm only", "minilm + bm25 (RRF)"):
                    baseline[row["config"]] = row
    except FileNotFoundError:
        pass

    header = f"{'configuration':32} {'docR@10':>8} {'docMRR':>8} {'parR@10':>8} {'parMRR':>8}"
    print("\n" + header)
    print("-" * len(header))

    def row(name, doc, par):
        print(f"{name:32} {doc['recall@10']:>8.3f} {doc['mrr@10']:>8.3f} "
              f"{par['recall@10']:>8.3f} {par['mrr@10']:>8.3f}")

    if "minilm only" in baseline:
        b = baseline["minilm only"]
        row("minilm, full text", b["docstring"], b["paraphrase"])
    row("minilm, cut input", report["sets"]["dense/docstring"],
        report["sets"]["dense/paraphrase"])
    if "minilm + bm25 (RRF)" in baseline:
        b = baseline["minilm + bm25 (RRF)"]
        row("minilm + bm25, full text", b["docstring"], b["paraphrase"])
    row("minilm + bm25, cut input", report["sets"]["hybrid/docstring"],
        report["sets"]["hybrid/paraphrase"])

    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
