# Response to the Previous Review

Every item kanungle raised on
[qdrant/demo-food-discovery#31](https://github.com/qdrant/demo-food-discovery/pull/31)
and [qdrant/demo-code-search#32](https://github.com/qdrant/demo-code-search/pull/32),
checked against this repository. His closing note on #31 was to run the
`neil-review` skill before submitting, which was done, and its own findings are
at the bottom.

Two of his items were open here when this audit started. Both are fixed.

## demo-food-discovery#31: The Eight Must-Fix Items

### 1. The Demo Names the Wrong Qdrant API

His finding: the hero said Discovery API while the code called
`recommend_groups()`. "This is our own demo teaching developers the wrong API
name."

**Here:** the product is **Qdrant Cloud Inference**, and the search is the
**Query API** (`points/query` with `prefetch` and `fusion`). Bare "Cloud
Inference" was corrected to the full product name in 15 places. The badge
tooltip names the two models from the API response rather than from a hardcoded
string, so it cannot drift from what actually answered.

### 2. False Claims in the "How It Works" Modal

His finding: four claims the code did not support, plus one that was only true
under a non-default strategy.

**Here:** the encoder swap made three claims false, and all three are corrected.

| was | now | why it was wrong |
|---|---|---|
| "Search Code by Meaning, **Not Keywords**" | "Search Code by Meaning, **and by Name**" | the demo runs BM25; keyword matching answers 0.980 recall@5 on the docstring set |
| "This demo runs **semantic** search" | "This demo runs **hybrid** search" | dense plus sparse, fused |
| badge: **Semantic** | badge: **Hybrid** | same claim, shown next to every result |
| "MiniLM reads the description, UniXcoder reads the code" | "mxbai reads the description, BM25 reads the identifiers" | neither model is in the build any more |

The two stat claims were checked against the live collection rather than carried
over: 14,604 functions (`code_type == "Function"`, counted) and 1,720 files.
Both hold.

### 3. The Description Says the Backend Is Unchanged When It Changed

His finding: reviewers skip a diff the description tells them to skip.

**Here:** this was open. An earlier draft of the PR description said the merge
step was "output identical" to the original. It is not, deliberately: the
re-sort is gone. The description now leads with four behavioural changes before
any of the hosting story.

### 4. The Declared Dependency Floor Cannot Run the New Code

His finding: `qdrant-client = "^1.6.0"` while the code needed 1.11, which
resolves fine for the author and breaks for anyone pinned.

**Here:** this was open, and it is the same class of gap. There are no Python or
client dependencies left to pin, but the *server* has a floor and nothing
declared it. The search is a two-leg `prefetch` with a `fusion` step, which the
Query API gained in **Qdrant 1.10**. Now stated in both the README prerequisites
and DEPLOY.md, with the note that Qdrant Cloud Inference is a separate
requirement: a 1.19 cluster with inference switched off still cannot embed a
query.

### 5. Deleting `.gitignore` Leaves `node_modules` Trackable

**Here:** root `.gitignore` covers `node_modules/`, `__pycache__/`, `*.pyc`,
`.env`, `.env.local`, `.vercel`, `.test-build/`, `data/` (with `!data/.keep` so
the directory the indexing scripts write into survives a fresh clone),
`indexer/.state/`, and `bench/results/*.log`. `frontend/.gitignore` is still
present.

### 6. The `vercel.json` Catch-All Rewrite Swallows the API

**Here:** the rewrite is `/((?!api/)(?!.*\.).*)`, which excludes the API prefix
and anything with a file extension. Verified against the deployment rather than
by reading it: `/api/search`, `/api/file` and `/api/health` all answer 200 in
production.

### 7. Search Failures Are Silent

His finding: `.error-state` was fully styled and never rendered, so a dead
backend looked like a working demo showing stale cards. He tested it by forcing
the API to fail.

**Here:** tested the same way. Pointing `QDRANT_NLU_COLLECTION` at a collection
that does not exist returns
`503 Search index is unavailable: a Qdrant collection is missing`, and the UI
renders the "Something Went Wrong" banner with the previous results cleared. The
API distinguishes a missing collection (503) from a query with no matches (200
and an empty list), which the old backend did not: it answered 200 either way,
and that is how the demo sat broken without anyone noticing.

### 8. The README No Longer Describes This App

His finding: the README credited a UI kit the PR deleted, and the screenshots
showed the old interface.

**Here:** rewritten around what the code now does. No stale screenshots exist,
because there are none: the only image is `images/architecture.svg`, redrawn for
this build and checked against the running system (two vendors, the model names,
the three collection sizes). The old `architecture-diagram.png` showed FastAPI
and Railway and was deleted rather than kept.

## demo-code-search#32: The Three Questions

**"Why are you gating the QDRANT_API_KEY to JWTs only?"** Not gated. The app
checks that a key is present when the URL is remote and nothing else. The only
JWT-shaped check left is a `note:` line in the indexing workflow, which logs and
carries on, because self-hosted keys are not JWTs.

**"Are payload indexes properly created with file_uploader.py?"** Yes, and
verified live rather than assumed. `indexer/build.py` creates a keyword index on
`path` with `wait=true`; the deployed collection reports
`{'path': {'data_type': 'keyword', 'points': 1720}}`. Without it, clusters with
strict mode on refuse to filter an unindexed field and the file viewer 500s on
every result while search keeps working, which reads like a frontend bug.

**"Has the README been updated enough?"** Rewritten, not patched. Every number
in it comes from a script in `bench/` that can be re-run, and the sections that
would have been the weakest, what this gives up and where it stops working, are
both there.

## neil-review Findings on This Repository

Run before submitting, as he asked. Fixed: Title Case on 18 headings, "vector
database" removed, US English, reciprocal rank fusion now credits Cormack et al.
2009 with the contribution stated as the measurement rather than the method, and
a "Where This Stops Working" section naming four boundaries.

The checker also reports 13 missing frontmatter fields and an absolute
`qdrant.tech` link. Those are `qdrant/landing_page` article rules and do not
apply to a repository README, so they are left alone.

## What Is Deliberately Different

The goal was a better demo on two vendors, not a byte-identical one. Four things
behave differently and each is either forced or measured:

1. **The encoder.** Forced. UniXcoder is not in the Qdrant Cloud Inference
   catalog and cannot be added.
2. **Result ordering.** Chosen. Removing the highlight-count re-sort moved
   docstring recall@1 from 0.620 to 0.887.
3. **The "Warming Up" fallback.** Gone with the backend that produced it, and it
   never fired in production: it needed a `data/structures.json` that was never
   in the deployed image. Confirmed against the live old demo, which reports no
   `mode` field on any query.
4. **Response caching.** The old build cached the query vector. Qdrant Cloud
   Inference never returns the vector, so the response is cached instead.

Everything else was held constant on purpose, and checked: the CSS file is
byte-identical, the per-result JSON keys are identical, the corpus is the same
1,720 files, and `indexed_commit` is the same `74f3e85b`, so every result link
resolves to the same code.
