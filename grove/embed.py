"""The novelty check's arithmetic: chunking, embedding through StoneSage's scoped garden route, cosine similarity.

Nothing here judges quality. A work is "close" to an earlier one when the cosine similarity of their vectors passes a threshold
(default 0.90, to be tuned on real output); the resident decides what to do about it (the design notes, "Novelty, not quality").
"""

import json
import math
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

CHUNK_CHARS = 900          # the embedder takes inputs under 950 characters
MAX_CHUNKS = 40            # a longer work is embedded from its first 40 chunks
BATCH = 16
DEFAULT_THRESHOLD = 0.90
DEFAULT_DUPLICATE_THRESHOLD = 0.98   # at or above this a piece is an exact repeat: refused outright, no variation_on override


class EmbedError(RuntimeError):
    pass


def chunk_text(text: str, limit: int = CHUNK_CHARS) -> List[str]:
    """Pieces of at most `limit` characters, cut at paragraph, sentence or word breaks where possible."""
    text = str(text or "").strip()
    out: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        while len(para) > limit:
            cut = max(para.rfind(". ", 0, limit), para.rfind("? ", 0, limit), para.rfind("! ", 0, limit))
            if cut < limit // 3:
                cut = para.rfind(" ", 0, limit)
            if cut < limit // 3:
                cut = limit - 1
            out.append(para[:cut + 1].strip())
            para = para[cut + 1:].strip()
        if para:
            out.append(para)
    merged: List[str] = []
    for piece in out:                                   # small paragraphs are joined so a poem is not 40 one-line inputs
        if merged and len(merged[-1]) + len(piece) + 1 <= limit:
            merged[-1] += " " + piece
        else:
            merged.append(piece)
    return merged[:MAX_CHUNKS]


def normalise(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else list(v)


def mean_vector(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            raise EmbedError("vectors of different sizes")
        for i, x in enumerate(v):
            acc[i] += x
    return normalise([x / len(vectors) for x in acc])


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


class HttpEmbedder:
    """POSTs {"input": [...]} to StoneSage's garden embeddings route with the garden model key; returns one vector per input."""

    def __init__(self, url: str, key: str, timeout: float = 60.0):
        url = url.rstrip("/")
        self.url = url if url.endswith("/embeddings") else url + "/embeddings"
        self._key, self.timeout = key, timeout

    def __call__(self, inputs: List[str]) -> List[List[float]]:
        req = urllib.request.Request(self.url, data=json.dumps({"input": inputs}).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._key})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            rows = sorted(data["data"], key=lambda d: d.get("index", 0))
            return [[float(x) for x in d["embedding"]] for d in rows]
        except urllib.error.HTTPError as e:
            raise EmbedError("embeddings HTTP %d" % e.code) from None
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
            raise EmbedError("the embeddings route did not answer") from None


def embed_text(embedder: Callable[[List[str]], List[List[float]]], text: str) -> List[float]:
    """One normalised vector for a whole text: the mean of its chunks' vectors."""
    chunks = chunk_text(text)
    if not chunks:
        raise EmbedError("nothing to embed")
    vectors: List[List[float]] = []
    for i in range(0, len(chunks), BATCH):
        part = chunks[i:i + BATCH]
        got = embedder(part)
        if len(got) != len(part):
            raise EmbedError("the embedder returned %d vectors for %d inputs" % (len(got), len(part)))
        vectors += [normalise(v) for v in got]
    return mean_vector(vectors)


def nearest(vector: List[float], known: Dict[str, List[float]]) -> Tuple[Optional[str], float]:
    """(work file, similarity) of the closest earlier work, or (None, 0.0)."""
    best, score = None, 0.0
    for name, v in known.items():
        s = cosine(vector, v)
        if best is None or s > score:
            best, score = name, s
    return best, round(score, 4)
