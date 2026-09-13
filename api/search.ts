import type { VercelRequest, VercelResponse } from "@vercel/node";

import { cacheGet, cacheKey, cacheSet } from "./_lib/cache.js";
import { DENSE_MODEL, INDEXED_COMMIT, SPARSE_MODEL } from "./_lib/config.js";
import { QdrantError } from "./_lib/qdrant.js";
import { search } from "./_lib/search.js";

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const query = typeof req.query.query === "string" ? req.query.query.trim() : "";
  if (!query) {
    res.status(400).json({ detail: "query is required" });
    return;
  }

  // The UI always shows five. A larger limit is allowed so a benchmark can ask
  // for a deeper cut without a second deployment, capped so a crawler cannot
  // turn one request into a full scan.
  const requested = Number(req.query.limit);
  const limit = Number.isFinite(requested) ? Math.min(Math.max(requested, 1), 20) : 5;

  // Time the work this function is responsible for: the round trip to Qdrant,
  // which now includes the embedding. Network time to the viewer is the
  // caller's, and reporting a number that moves with their connection would
  // make it meaningless. The UI shows this rather than asserting a figure, so
  // it cannot go stale.
  const started = Date.now();
  const key = cacheKey(query, limit);
  const hit = cacheGet<Awaited<ReturnType<typeof search>>>(key);
  try {
    const outcome = hit ?? (await search(query, limit));
    if (!hit) cacheSet(key, outcome);
    res.status(200).json({
      result: outcome.results,
      latency_ms: Date.now() - started,
      indexed_commit: INDEXED_COMMIT,
      // Tokens the cluster billed for this search. Zero on a cache hit, because
      // nothing was embedded, which is what makes the cache visible in the
      // numbers rather than something to take on trust.
      inference_tokens: hit ? 0 : outcome.tokens,
      cached: Boolean(hit),
      // Named here rather than in the UI so the badge cannot drift from what
      // actually answered the query.
      models: { dense: DENSE_MODEL, sparse: SPARSE_MODEL },
    });
  } catch (err) {
    if (err instanceof QdrantError && err.status === 404) {
      // A missing collection is a deploy state, not a query with no matches.
      // The old backend answered 200 with an empty list here, which is how the
      // demo sat broken without anyone noticing.
      res.status(503).json({
        detail:
          "Search index is unavailable: a Qdrant collection is missing. " +
          "Run indexer/build.py to populate it.",
      });
      return;
    }
    const status = err instanceof QdrantError ? err.status : 500;
    res.status(status).json({ detail: err instanceof Error ? err.message : String(err) });
  }
}
