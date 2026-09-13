"""Stdlib-only Qdrant REST access for the indexer.

There is no qdrant-client dependency here for the same reason there is no torch:
the indexer's whole job is now to post text and let the cluster embed it, and a
client library would be the largest thing in a repository that otherwise has no
runtime at all. It also keeps the indexer runnable on any Python without a wheel
hunt, which matters because the machine this was written on only has 3.14.

Connections are pooled per thread and reused. A full run is several thousand
requests; opening a TCP and TLS connection for each one produced sporadic
`[WinError 10054] connection forcibly closed` failures that looked like the
cluster misbehaving and were the client's own connection churn.
"""

import http.client
import json
import os
import random
import threading
import time
import urllib.parse

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

if QDRANT_URL.startswith("https://") and not QDRANT_API_KEY:
    raise SystemExit(
        f"QDRANT_URL is remote ({QDRANT_URL}) but QDRANT_API_KEY is not set. "
        "Copy .env.example to .env and fill both in."
    )

_PARSED = urllib.parse.urlparse(QDRANT_URL)
_HTTPS = _PARSED.scheme == "https"
_HOST = _PARSED.hostname
_PORT = _PARSED.port or (443 if _HTTPS else 6333)

# Retried statuses. 429 is the inference rate limit and 5xx is the cluster
# briefly refusing work; both are normal over a run of this length and neither
# should abort hours of indexing.
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 8

_local = threading.local()


class QdrantError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def _connection(fresh=False):
    """One keep-alive connection per thread, reopened on demand."""
    conn = getattr(_local, "conn", None)
    if conn is not None and fresh:
        conn.close()
        conn = None
    if conn is None:
        cls = http.client.HTTPSConnection if _HTTPS else http.client.HTTPConnection
        conn = cls(_HOST, _PORT, timeout=300)
        _local.conn = conn
    return conn


def call(method, path, body=None, timeout=300):
    """One REST call with backoff. Returns (result, usage)."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
    last = None

    for attempt in range(MAX_ATTEMPTS):
        # A connection that failed is not reused. Retrying down a half-closed
        # socket fails identically every time, which is how one blip turns into
        # a run that never recovers.
        conn = _connection(fresh=attempt > 0)
        try:
            conn.request(method, path, body=data, headers=headers)
            response = conn.getresponse()
            raw = response.read()

            if response.status in RETRY_STATUS:
                last = QdrantError(
                    f"{method} {path} -> {response.status}: {raw.decode()[:400]}",
                    response.status,
                )
            else:
                payload = json.loads(raw)
                if "result" not in payload:
                    raise QdrantError(
                        f"{method} {path} -> {response.status}: {json.dumps(payload)[:400]}",
                        response.status,
                    )
                return payload["result"], payload.get("usage", {})
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
            last = QdrantError(f"{method} {path} -> {type(exc).__name__}: {exc}")

        # Exponential backoff with jitter, floored: pure full jitter can pick a
        # delay of almost zero, which retries straight into the same problem.
        time.sleep(min(30.0, 2**attempt) * (0.5 + 0.5 * random.random()))

    raise last


def tokens_by_model(usage):
    models = (usage or {}).get("inference", {}).get("models", {})
    return {name: m.get("tokens", 0) for name, m in models.items()}


def scroll(collection, batch=512, with_vector=False):
    """Yield every payload in a collection. Read-only."""
    offset = None
    while True:
        body = {"limit": batch, "with_payload": True, "with_vector": with_vector}
        if offset is not None:
            body["offset"] = offset
        result, _usage = call("POST", f"/collections/{collection}/points/scroll", body)
        for point in result["points"]:
            yield point["payload"]
        offset = result.get("next_page_offset")
        if offset is None:
            return
