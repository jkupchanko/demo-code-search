/**
 * Minimal Qdrant REST client.
 *
 * The old backend carried torch, transformers and sentence-transformers so it
 * could turn a query into a vector before asking Qdrant anything. Cloud
 * Inference does that inside the cluster, so the client's entire job is to post
 * JSON. That is small enough that a dependency would be the larger half of it,
 * and keeping it dependency-free is what lets these functions cold-start in
 * milliseconds instead of loading a model.
 */

const RAW_URL = process.env.QDRANT_URL ?? "";
const API_KEY = process.env.QDRANT_API_KEY ?? "";

if (!RAW_URL) {
  throw new Error("QDRANT_URL is not set");
}
if (RAW_URL.startsWith("https://") && !API_KEY) {
  // Fail loudly at module load rather than returning 500s per request. A
  // missing variable on a remote cluster is a deploy problem, and surfacing it
  // as "unauthorized" on every search sends whoever debugs it the wrong way.
  throw new Error(`QDRANT_URL is remote (${RAW_URL}) but QDRANT_API_KEY is not set`);
}

const BASE = RAW_URL.replace(/\/+$/, "");

/** Milliseconds before a Qdrant call is abandoned. */
const TIMEOUT_MS = Number(process.env.QDRANT_TIMEOUT_MS ?? 15000);

export type InferenceUsage = {
  inference?: { models?: Record<string, { tokens?: number }> };
};

export class QdrantError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "QdrantError";
  }
}

export async function post<T>(path: string, body: unknown): Promise<{ result: T; usage?: InferenceUsage }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: { "api-key": API_KEY, "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    const aborted = err instanceof Error && err.name === "AbortError";
    throw new QdrantError(
      aborted ? `Qdrant did not answer within ${TIMEOUT_MS}ms` : String(err),
      aborted ? 504 : 502,
    );
  } finally {
    clearTimeout(timer);
  }

  const text = await response.text();
  let payload: { result?: T; status?: unknown; usage?: InferenceUsage };
  try {
    payload = JSON.parse(text);
  } catch {
    throw new QdrantError(`Qdrant returned non-JSON (${response.status}): ${text.slice(0, 200)}`, 502);
  }

  if (!response.ok || payload.result === undefined) {
    const detail =
      typeof payload.status === "object" && payload.status !== null && "error" in payload.status
        ? String((payload.status as { error: unknown }).error)
        : text.slice(0, 300);
    throw new QdrantError(detail, response.status === 404 ? 404 : 502);
  }

  return { result: payload.result, usage: payload.usage };
}

/** Total billable inference tokens across every model in one response. */
export function tokensUsed(usage?: InferenceUsage): number {
  const models = usage?.inference?.models ?? {};
  return Object.values(models).reduce((sum, m) => sum + (m.tokens ?? 0), 0);
}
