import type { VercelRequest, VercelResponse } from "@vercel/node";

import {
  CODE_COLLECTION,
  DENSE_MODEL,
  FILE_COLLECTION,
  NLU_COLLECTION,
  SPARSE_MODEL,
} from "./_lib/config.js";
import { post, QdrantError } from "./_lib/qdrant.js";

/**
 * Liveness plus the two things that actually break: a collection that is not
 * there, and point counts that say an indexing run stopped halfway. Reporting
 * only "ok" would have passed on every day the demo was broken.
 */
export default async function handler(_req: VercelRequest, res: VercelResponse) {
  const names = {
    code: CODE_COLLECTION,
    signatures: NLU_COLLECTION,
    files: FILE_COLLECTION,
  };

  const collections: Record<string, number | string> = {};
  let ok = true;

  await Promise.all(
    Object.entries(names).map(async ([label, name]) => {
      try {
        const { result } = await post<{ count: number }>(
          `/collections/${name}/points/count`,
          { exact: false },
        );
        collections[label] = result.count;
        if (result.count === 0) ok = false;
      } catch (err) {
        collections[label] = err instanceof QdrantError ? `error: ${err.message}` : "error";
        ok = false;
      }
    }),
  );

  res.status(ok ? 200 : 503).json({
    status: ok ? "ok" : "degraded",
    collections,
    models: { dense: DENSE_MODEL, sparse: SPARSE_MODEL },
  });
}
