"""Shared REST helpers for the benchmarks.

Deliberately stdlib-only. The benchmark has to be runnable by anyone reviewing
the numbers, on a machine that has nothing installed, and the whole point of
moving to Qdrant Cloud Inference is that the client no longer needs a model runtime.
Pulling in qdrant-client here would hide how little the client actually does.

Connections are kept alive and reused. The first version opened a fresh TCP and
TLS connection per request, and a full run is a few thousand requests: that
churn produced sporadic `[WinError 10054] connection forcibly closed` failures
that looked like the cluster misbehaving and were this script's own fault. It
also put a TLS handshake inside every latency measurement, which is not
something a real client pays on every search.
"""

import http.client
import json
import os
import random
import threading
import time
import urllib.parse

QDRANT_URL = os.environ["QDRANT_URL"].rstrip("/")
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

_PARSED = urllib.parse.urlparse(QDRANT_URL)
_HTTPS = _PARSED.scheme == "https"
_HOST = _PARSED.hostname
_PORT = _PARSED.port or (443 if _HTTPS else 6333)

# Every model here runs inside the Qdrant cluster. Nothing in this file talks to
# a second vendor, which is the property the rebuild exists to have.
DENSE = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 384),
    "mxbai": ("mixedbread-ai/mxbai-embed-large-v1", 1024),
}
SPARSE = {
    "bm25": "Qdrant/bm25",
    "splade": "prithivida/Splade_PP_en_v1",
}

# Retried statuses. 429 is the inference rate limit; the 5xxs are the cluster
# briefly refusing work. Both happen over a run of a few thousand requests, and
# neither is a result - a benchmark that dies at row six and reports nothing is
# worse than one that waits a second.
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 8


class QdrantError(RuntimeError):
    pass


class Pool:
    """Keep-alive connections to one host, one per thread.

    Used for the Qdrant cluster below and, in latency.py and quality.py, for the
    deployed demo. Same reasoning both times: a TLS handshake per request is not
    what a browser pays, so including one in every sample would report a number
    nobody experiences.
    """

    def __init__(self, base_url, timeout=300):
        parsed = urllib.parse.urlparse(base_url)
        self.https = parsed.scheme != "http"
        self.host = parsed.hostname
        self.port = parsed.port or (443 if self.https else 80)
        self.timeout = timeout
        self._local = threading.local()

    def connection(self, fresh=False):
        conn = getattr(self._local, "conn", None)
        if conn is not None and fresh:
            conn.close()
            conn = None
        if conn is None:
            cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
            conn = cls(self.host, self.port, timeout=self.timeout)
            self._local.conn = conn
        return conn

    def get(self, path, headers=None):
        """One GET, no retry. Returns (status, body_bytes, elapsed_ms).

        No retry on purpose: this measures a deployment, and silently retrying a
        failed request would turn an error rate into a latency figure.
        """
        conn = self.connection()
        started = time.perf_counter()
        try:
            conn.request("GET", path, headers=headers or {})
            response = conn.getresponse()
            raw = response.read()
            return response.status, raw, (time.perf_counter() - started) * 1000
        except (OSError, http.client.HTTPException) as exc:
            self.connection(fresh=True)
            return 0, str(exc).encode(), (time.perf_counter() - started) * 1000


_pool = Pool(QDRANT_URL)


def _connection(fresh=False):
    return _pool.connection(fresh=fresh)


def call(method: str, path: str, body=None, timeout=300):
    """One REST call, retried on transient failures.

    Returns (result, elapsed_ms, usage). elapsed_ms times the successful attempt
    only: a latency figure that included backoff would measure this script's
    retry policy rather than the search.
    """
    data = json.dumps(body).encode() if body is not None else None
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
    last = None

    for attempt in range(MAX_ATTEMPTS):
        # A connection that failed is not reused. Retrying a request down a
        # half-closed socket fails the same way every time, which is how a
        # transient blip turns into a run that never recovers.
        conn = _connection(fresh=attempt > 0)
        started = time.perf_counter()
        try:
            conn.request(method, path, body=data, headers=headers)
            response = conn.getresponse()
            raw = response.read()
            elapsed = (time.perf_counter() - started) * 1000

            if response.status in RETRY_STATUS:
                last = QdrantError(f"{method} {path} -> {response.status}: {raw.decode()[:400]}")
            else:
                payload = json.loads(raw)
                if "result" not in payload:
                    raise QdrantError(
                        f"{method} {path} -> {response.status}: {json.dumps(payload)[:400]}"
                    )
                return payload["result"], elapsed, payload.get("usage", {})
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            last = QdrantError(f"{method} {path} -> {type(exc).__name__}: {exc}")

        # Exponential backoff with jitter, but with a floor: pure full jitter
        # can pick a delay of almost zero, which retries into the same problem.
        time.sleep(min(30.0, 2**attempt) * (0.5 + 0.5 * random.random()))

    raise last


def target_headers() -> dict:
    """Headers for requests to a deployed demo.

    A Vercel deployment with Deployment Protection on answers 302 to its SSO
    page instead of serving the API, which a benchmark would otherwise record as
    a fast response to nothing. Setting VERCEL_PROTECTION_BYPASS to the
    project's automation bypass secret lets the measurement through while the
    URL stays closed to everyone else.
    """
    headers = {"accept": "application/json"}
    secret = os.environ.get("VERCEL_PROTECTION_BYPASS")
    if secret:
        headers["x-vercel-protection-bypass"] = secret
        # Stops the bypass from setting a cookie that would make later requests
        # take a different path through the edge than the first one.
        headers["x-vercel-set-bypass-cookie"] = "false"
    return headers


def tokens_of(usage: dict) -> int:
    """Billable tokens reported by Qdrant Cloud Inference for one call.

    Qdrant returns this per model on every request that used inference, which is
    the only honest way to price a full index run: measure a sample, multiply.
    BM25 is computed in-engine and never appears here, which is itself a result.
    """
    models = (usage or {}).get("inference", {}).get("models", {})
    return sum(m.get("tokens", 0) for m in models.values())


def tokens_by_model(usage: dict) -> dict:
    models = (usage or {}).get("inference", {}).get("models", {})
    return {name: m.get("tokens", 0) for name, m in models.items()}


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(round((p / 100) * (len(ordered) - 1))), len(ordered) - 1)
    return round(ordered[idx], 1)
