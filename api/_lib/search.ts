import {
  CODE_COLLECTION,
  DENSE_MODEL,
  DENSE_VECTOR,
  NLU_COLLECTION,
  PREFETCH_LIMIT,
  SPARSE_MODEL,
  SPARSE_VECTOR,
} from "./config.js";
import { mergeSearchResults, type CodeHit, type NluHit } from "./merge.js";
import { post, tokensUsed } from "./qdrant.js";

type Scored<T> = { id: string | number; score: number; payload: T };

/**
 * A hybrid query: dense and sparse legs, fused server-side with reciprocal rank
 * fusion.
 *
 * Both legs carry the query as text rather than as a vector. The cluster embeds
 * it, searches, and fuses in one round trip, which is why this repository has no
 * model runtime in it at all. The dense leg costs a handful of inference tokens;
 * BM25 costs none, because the engine derives it from collection statistics.
 */
function hybridBody(query: string, limit: number, withPayload: true | string[]) {
  return {
    prefetch: [
      {
        query: { text: query, model: DENSE_MODEL },
        using: DENSE_VECTOR,
        limit: PREFETCH_LIMIT,
      },
      {
        query: { text: query, model: SPARSE_MODEL },
        using: SPARSE_VECTOR,
        limit: PREFETCH_LIMIT,
      },
    ],
    query: { fusion: "rrf" },
    limit,
    with_payload: withPayload,
  };
}

export type SearchOutcome = {
  results: NluHit[];
  tokens: number;
  timings: { nlu_ms: number; code_ms: number };
};

export async function search(query: string, limit = 5, codeLimit = 20): Promise<SearchOutcome> {
  // The two searches share nothing but the query string, so they go out
  // together. Serialising them made every search wait for both round trips end
  // to end, which on a cluster in another region is most of the response.
  const nluStarted = Date.now();
  const nluCall = post<{ points: Scored<NluHit>[] }>(
    `/collections/${NLU_COLLECTION}/points/query`,
    hybridBody(query, limit, true),
  ).then((r) => ({ ...r, ms: Date.now() - nluStarted }));

  const codeStarted = Date.now();
  const codeCall = post<{ points: Scored<CodeHit>[] }>(
    `/collections/${CODE_COLLECTION}/points/query`,
    hybridBody(query, codeLimit, ["start_line", "end_line", "file"]),
  ).then((r) => ({ ...r, ms: Date.now() - codeStarted }));

  const [nlu, code] = await Promise.all([nluCall, codeCall]);

  return {
    results: mergeSearchResults(
      code.result.points.map((p) => p.payload),
      nlu.result.points.map((p) => p.payload),
    ),
    tokens: tokensUsed(nlu.usage) + tokensUsed(code.usage),
    timings: { nlu_ms: nlu.ms, code_ms: code.ms },
  };
}
