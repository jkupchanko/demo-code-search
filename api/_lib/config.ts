/**
 * One place for the names of things.
 *
 * Every model here is served from inside the Qdrant cluster. External providers
 * (openai/, cohere/, jinaai/, openrouter/) are reachable through Cloud
 * Inference too, but each one is a second vendor with a second API key, which is
 * the thing this rebuild exists to remove. If you change a model, change it to
 * another in-cluster one.
 */

export const CODE_COLLECTION = process.env.QDRANT_CODE_COLLECTION ?? "code-snippets-cloud";
export const NLU_COLLECTION = process.env.QDRANT_NLU_COLLECTION ?? "code-signatures-cloud";
export const FILE_COLLECTION = process.env.QDRANT_FILE_COLLECTION ?? "code-files-cloud";

/**
 * Dense encoder. Chosen by bench/bakeoff.py; see the table in the README.
 *
 * mxbai beat all-MiniLM-L6-v2 on both query sets and beat it decisively on the
 * one that matters for a demo: 13.3% of natural-language paraphrase queries put
 * the right function first, against 3.5% for MiniLM.
 */
export const DENSE_MODEL =
  process.env.QDRANT_DENSE_MODEL ?? "mixedbread-ai/mxbai-embed-large-v1";
export const DENSE_VECTOR = "dense";

/**
 * Sparse encoder. BM25 is computed by the engine from collection statistics, so
 * unlike every other model in the catalog it bills no inference tokens - the
 * sparse leg of a hybrid query is free.
 */
export const SPARSE_MODEL = process.env.QDRANT_SPARSE_MODEL ?? "Qdrant/bm25";
export const SPARSE_VECTOR = "sparse";

/** Candidates each leg contributes to the fusion before the final cut. */
export const PREFETCH_LIMIT = Number(process.env.PREFETCH_LIMIT ?? 100);

/**
 * Commit of qdrant/qdrant the collections were built from. Result links carry
 * line numbers, and resolving them against a moving `master` quietly points at
 * the wrong lines as the source changes.
 */
export const INDEXED_COMMIT = process.env.INDEXED_COMMIT ?? "master";
