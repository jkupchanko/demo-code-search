import type { VercelRequest, VercelResponse } from "@vercel/node";

import { FILE_COLLECTION } from "./_lib/config.js";
import { post, QdrantError } from "./_lib/qdrant.js";

type FilePayload = {
  path: string;
  code: string[];
  startline?: number;
  endline?: number;
};

/**
 * The file viewer. A filtered scroll on `path` and nothing else - this
 * collection holds no vectors, it is Qdrant used as the demo's only datastore
 * so that there is no second thing to deploy.
 *
 * `path` must have a keyword payload index. Clusters with strict mode on refuse
 * to filter an unindexed field, and without it every result anyone clicks
 * returns a 500 while search itself keeps working, which reads like a frontend
 * bug. indexer/build.py creates it.
 */
export default async function handler(req: VercelRequest, res: VercelResponse) {
  const path = typeof req.query.path === "string" ? req.query.path : "";
  if (!path) {
    res.status(400).json({ detail: "path is required" });
    return;
  }

  try {
    const { result } = await post<{ points: { payload: FilePayload }[] }>(
      `/collections/${FILE_COLLECTION}/points/scroll`,
      {
        filter: { must: [{ key: "path", match: { value: path } }] },
        limit: 5,
        with_payload: true,
        with_vector: false,
      },
    );
    res.status(200).json({ result: result.points.map((p) => p.payload) });
  } catch (err) {
    const status = err instanceof QdrantError ? err.status : 500;
    res.status(status).json({ detail: err instanceof Error ? err.message : String(err) });
  }
}
