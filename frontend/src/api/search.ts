import { api } from "./axios";
import { SEARCH_URL } from "./constants";

export type SearchResponse = {
  /**
   * Server-side time to answer, in milliseconds. The embedding happens inside
   * the Qdrant cluster now, so this covers the whole round trip from the
   * function to Qdrant and back. Excludes the viewer's own network. Optional,
   * since an older backend will not send it.
   */
  latency_ms?: number;
  /**
   * Commit of qdrant/qdrant the index was built from. Result links carry line
   * numbers, so they have to resolve against this rather than a moving branch.
   * Optional: an older backend will not send it, and links fall back to master.
   */
  indexed_commit?: string;
  /**
   * Inference tokens the cluster billed for this search. Small - a query is a
   * handful of tokens, and the BM25 leg costs none - but it is the number that
   * makes the running cost of the demo checkable rather than asserted.
   */
  inference_tokens?: number;
  /**
   * True when this instance had already answered the same query and replayed
   * the stored response. The index is a snapshot, so a replay is the same
   * answer, not a stale one.
   */
  cached?: boolean;
  /**
   * Which models answered. Sent by the backend rather than hardcoded here, so
   * the badge cannot claim one encoder while the index was built with another.
   */
  models?: {
    dense: string;
    sparse: string;
  };
  result: {
    code_type: string;
    context: {
      file_name: string;
      file_path: string;
      module: string;
      snippet: string;
      /** Null for free functions, which are not attached to a struct. */
      struct_name: string | null;
    };
    docstring: string | null;
    line: number;
    line_from: number;
    line_to: number;
    name: string;
    signature: string;
    /**
     * Line ranges where the two searches agreed, used to highlight inside the
     * snippet. Only present when a result's file also came back from the code
     * snippet search, which is a minority of them - so this is genuinely
     * optional and was previously typed as though it always arrived.
     */
    sub_matches?: {
      overlap_from: number;
      overlap_to: number;
    }[];
  }[];
};

export const getSearchResult = (query: string) =>
  api.get<SearchResponse>(SEARCH_URL, { params: { query } });
