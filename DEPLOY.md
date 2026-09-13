# Deploying

Two accounts, one deployment, two environment variables.

| Piece | Where | Cost |
|---|---|---|
| Frontend and API (one Vercel project) | Vercel | Free (Hobby) |
| Vector search engine and embedding models | Qdrant Cloud | Free tier (1 GB) |

The previous build needed a third: Railway, running a container with torch,
transformers and sentence-transformers in it, because the backend embedded the
query itself. Qdrant Cloud Inference does that inside the cluster, so there is
nothing left for that container to do. The API is four small TypeScript
functions that post JSON.

---

## 1. Qdrant Cloud

1. Create a cluster at https://cloud.qdrant.io. The free 1 GB tier holds this
   dataset with room to spare.
2. Open the cluster, go to the **Inference** tab, and check it is enabled.
   Clusters created after 7 July 2025 have it on already. Enabling it on an
   older cluster **restarts the cluster**, so do that before you point anything
   at it, not after.
3. Copy the cluster URL (including `:6333`) and create an API key.

**Server minimum: Qdrant 1.10.** The search is a `prefetch` with two legs and a
`fusion` step, which the Query API gained in 1.10. An older server rejects the
request outright rather than degrading, so this fails loudly rather than
quietly. Qdrant Cloud Inference is a separate requirement on top of the version:
a 1.19 cluster with inference switched off still cannot embed the query.

Check inference is actually answering before you spend hours indexing:

```bash
curl -sS -X POST "$QDRANT_URL/collections/does-not-exist/points/query" \
  -H "api-key: $QDRANT_API_KEY" -H 'Content-Type: application/json' \
  -d '{"query":{"text":"probe","model":"sentence-transformers/all-MiniLM-L6-v2"},"limit":1}'
```

`Collection 'does-not-exist' doesn't exist` is the answer you want: the request
reached the engine, which means the model resolved. `Unsupported model` or
`Expected some form of vector` means inference is off for this cluster.

## 2. Build the Index

Either build from source, which needs Rust, rust-analyzer and Docker:

```bash
export QDRANT_URL="https://your-cluster-id.region.aws.cloud.qdrant.io:6333"
export QDRANT_API_KEY="..."
bash tools/download_and_index.sh
```

Or, if you are migrating from an existing deployment of the older demo, copy the
corpus straight out of its collections and re-embed it:

```bash
python indexer/build.py --source qdrant
```

That reads `code-snippets-unixcoder`, `code-signatures` and `code-files` and
writes `code-snippets-cloud`, `code-signatures-cloud` and `code-files-cloud`.
The old collections are only ever read, so the old demo keeps working while this
one is built, and both can run side by side until you switch the domain over.

Runs are resumable. If one dies at hour two, run it again and it skips the
batches already acknowledged instead of paying to embed them twice.

Note the commit the index was built from. `tools/download_and_index.sh` prints it
at the end; it goes into `INDEXED_COMMIT` below, and without it every result link
resolves against a moving `master` and eventually points at the wrong lines.

## 3. Vercel

1. **Add New → Project**, import this repo, leave the root directory at the
   repository root. [vercel.json](vercel.json) builds the frontend into
   `frontend/dist` and picks up the functions in [`api/`](api).
2. Add the environment variables:

   | Variable | Value |
   |---|---|
   | `QDRANT_URL` | cluster URL including `:6333` |
   | `QDRANT_API_KEY` | cluster API key |
   | `INDEXED_COMMIT` | the qdrant/qdrant SHA the collections were built from |

   Everything else has a working default; see [.env.example](.env.example).
3. Deploy.

There is no `VITE_API_URL` and no `CORS_ORIGINS`, because the app and the API
are served from one origin. If you find yourself adding either one back, the
deployment has split in two again.

## 4. Verify

```bash
curl -s https://<your-deployment>/api/health | python -m json.tool
```

It reports the point count of all three collections and the models in use, and
answers 503 if any collection is missing or empty. A bare `{"status":"ok"}`
would have passed on every day the old demo sat broken, so it does not do that.

Then measure it, rather than assuming:

```bash
python bench/latency.py --target https://<your-deployment>
python bench/quality.py --target https://<your-deployment>
```

## Local Development

```bash
cp .env.example .env     # fill in QDRANT_URL and QDRANT_API_KEY
npm install
npx vercel dev
```

`.env`, not `.env.local`: for a project that has not been linked with
`vercel link`, the CLI reads `.env` and leaves `.env.local` alone, and the
functions come up throwing `QDRANT_URL is not set`. Both are gitignored.

There is deliberately no `dev` script in `package.json`: `vercel dev` runs the
project's own `dev` script, so defining one as `vercel dev` makes it invoke
itself and refuse to start.

`vercel dev` serves the frontend and the functions together on
http://127.0.0.1:3000, which is the same shape as production. If you would rather
have Vite's hot reload, run `npm --prefix frontend run dev` alongside it; the dev
server proxies `/api` to port 3000.

## Troubleshooting

**Search returns 503 "a Qdrant collection is missing".** The index was not
built, or `QDRANT_*_COLLECTION` points at a name that does not exist. `/api/health`
names which one.

**Upserts fail with `Unsupported model`.** Qdrant Cloud Inference is off for the
cluster, or the model name is misspelled. Names are case-insensitive on the wire
but must otherwise match the Inference tab exactly.

**Upserts fail with `Vector dimension error`.** The
collection was created for one dense model and is being filled with another.
Rebuild it with `--fresh`, or set `QDRANT_DENSE_MODEL` back.

**File viewer 500s on every result.** The `path` payload index is missing.
`indexer/build.py` creates it; clusters with strict mode on refuse to filter an
unindexed field, and search keeps working meanwhile, which makes this look like a
frontend bug.

**A function times out at 20 seconds.** That is `maxDuration` in
[vercel.json](vercel.json), and hitting it means Qdrant is not answering, not
that the search is slow. Check the cluster.
