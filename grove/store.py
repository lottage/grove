"""The garden's files: residents (soul folders), their works, drafts, journal, wishes, session transcripts, window snapshots.

    <data dir>/souls/<id>/manifest.json persona.md journal.md wishes.jsonl   (top-level files = what stonesage/soul.py exports)
                          works/<n>-<slug>.md + .json   drafts/<session>.md   sessions/<session>.jsonl (append-only)
                          vectors/<work file>.json (embeddings for the novelty check)   comments.jsonl (users' comments and ratings)
    <data dir>/snapshots/<kind>.json    plain-text window snapshots StoneSage pushes; `household-terms` is the egress scan's word list

Standard library only. Session transcripts are only ever appended to; nothing here edits or deletes one.
"""

import json
import os
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

from . import texts

AGENT_ID = re.compile(r"^[a-z0-9-]{1,64}$")          # same rule as stonesage/soul.py
SESSION_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{4}$")
KIND = re.compile(r"^[a-z0-9_-]{1,40}$")
FORMAT_VERSION = 1
MAX_SNAPSHOT_CHARS = 64 * 1024
POST_ID = re.compile(r"^([a-z0-9-]{1,64}):(work|journal|wish|draft|session):([A-Za-z0-9._-]{1,80})$")
TERMS_KIND = "household-terms"
WINDOW_KINDS = ("weather", "cameras", "book", "headlines", "computer")     # what look_out_window can show
MAX_COMMENT_CHARS = 2000
MAX_AGENT_COMMENT_CHARS = 1000     # a resident's comment on another resident's work
MODELS_KIND = "models"             # the snapshot kind that carries the model catalogue (JSON text); not a window item
REQUEST_ID = re.compile(r"^([a-z0-9-]{1,64})\.r([0-9]{5,})$")
MAX_PENDING_REQUESTS = 3           # model requests a resident may have waiting at once
WISH_KINDS = (None, "wish", "model_request")      # lines of wishes.jsonl that are shown as wishes; the rest are decisions and receipts
MAX_WEB_LOG = 40
_JOURNAL_MARK = re.compile(r"^<!-- entry ts=([0-9.]+) session=([A-Za-z0-9_-]*) -->$", re.M)


class StoreError(ValueError):
    pass


def _write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return (s or "untitled")[:50]


class Store:
    def __init__(self, data_dir: str, clock=time.time):
        self.data_dir = data_dir
        self.clock = clock
        self.lock = threading.RLock()
        os.makedirs(os.path.join(data_dir, "souls"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "snapshots"), exist_ok=True)

    # ---- residents ---------------------------------------------------------------------------
    def root(self, resident: str) -> str:
        if not isinstance(resident, str) or not AGENT_ID.match(resident):
            raise StoreError("a resident id is 1-64 characters of a-z, 0-9 and '-'")
        return os.path.join(self.data_dir, "souls", resident)

    def exists(self, resident: str) -> bool:
        return os.path.isfile(os.path.join(self.root(resident), "manifest.json"))

    def create_resident(self, resident: str) -> Dict[str, Any]:
        """A new resident: no name, a minimal neutral persona, no subjects. An existing one is returned unchanged."""
        with self.lock:
            root = self.root(resident)
            if self.exists(resident):
                return self.manifest(resident)
            for sub in ("works", "drafts", "sessions"):
                os.makedirs(os.path.join(root, sub), exist_ok=True)
            manifest = {"id": resident, "name": resident, "named": False, "format": FORMAT_VERSION,
                        "created": round(self.clock(), 3), "persona_version": texts.PERSONA_VERSION,
                        "persona_seed": texts.sha256_text(texts.PERSONA)}
            _write(os.path.join(root, "persona.md"), texts.PERSONA + "\n")
            for name in ("journal.md", "wishes.jsonl"):
                open(os.path.join(root, name), "a", encoding="utf-8").close()
            self._save_manifest(resident, manifest)
            return manifest

    def manifest(self, resident: str) -> Dict[str, Any]:
        with open(os.path.join(self.root(resident), "manifest.json"), encoding="utf-8-sig") as f:
            return json.load(f)

    def _save_manifest(self, resident: str, manifest: Dict[str, Any]) -> None:
        _write(os.path.join(self.root(resident), "manifest.json"), json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    def persona(self, resident: str) -> str:
        with open(os.path.join(self.root(resident), "persona.md"), encoding="utf-8-sig") as f:
            return f.read().strip()

    def set_paused(self, resident: str, paused: bool) -> Dict[str, Any]:
        with self.lock:
            if not self.exists(resident):
                raise StoreError("no such resident")
            m = self.manifest(resident)
            m["paused"] = bool(paused)
            self._save_manifest(resident, m)
            return m

    def is_paused(self, resident: str) -> bool:
        try:
            return bool(self.manifest(resident).get("paused")) if self.exists(resident) else False
        except (OSError, ValueError):
            return False

    def residents_info(self) -> List[Dict[str, Any]]:
        out = []
        for rid in self.residents():
            m = self.manifest(rid)
            out.append({"id": rid, "name": m.get("name", rid), "named": bool(m.get("named")), "paused": bool(m.get("paused")),
                        "created": m.get("created"), "works": len(self.list_works(rid))})
        return out

    def name_self(self, resident: str, name: str) -> Dict[str, Any]:
        """Only while the resident has no name, and only once. Returns {ok, ...}."""
        name = " ".join(str(name or "").split())
        if not 1 <= len(name) <= 40:
            return {"ok": False, "error": "a name is 1-40 characters"}
        with self.lock:
            m = self.manifest(resident)
            if m.get("named"):
                return {"ok": False, "error": "you already have a name"}
            m.update({"name": name, "named": True, "named_at": round(self.clock(), 3)})
            self._save_manifest(resident, m)
        return {"ok": True, "name": name}

    # ---- works, drafts, journal, wishes --------------------------------------------------------
    def list_works(self, resident: str) -> List[Dict[str, Any]]:
        folder = os.path.join(self.root(resident), "works")
        out = []
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            if name.endswith(".json"):
                try:
                    with open(os.path.join(folder, name), encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["file"] = name[:-5] + ".md"
                    out.append(meta)
                except (OSError, ValueError):
                    continue
        return sorted(out, key=lambda m: m.get("at", 0))

    def find_work(self, resident: str, title: str) -> Optional[Dict[str, Any]]:
        want = " ".join(str(title or "").split()).lower()
        for meta in reversed(self.list_works(resident)):
            if " ".join(str(meta.get("title", "")).split()).lower() == want:
                return meta
        return None

    def read_work_text(self, resident: str, meta: Dict[str, Any]) -> str:
        with open(os.path.join(self.root(resident), "works", meta["file"]), encoding="utf-8") as f:
            return f.read()

    def add_work(self, resident: str, title: str, form: str, text: str, session: str, model: str,
                 extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Keep a finished work. `extra` (novelty score, variation_on) goes into its metadata. Returns the metadata plus `file`."""
        with self.lock:
            folder = os.path.join(self.root(resident), "works")
            os.makedirs(folder, exist_ok=True)
            n = len([f for f in os.listdir(folder) if f.endswith(".json")]) + 1
            base = "%04d-%s" % (n, slug(title))
            meta = {"title": title.strip(), "form": form.strip(), "session": session, "model": model,
                    "words": len(text.split()), "at": round(self.clock(), 3)}
            meta.update(extra or {})
            _write(os.path.join(folder, base + ".md"), text)
            _write(os.path.join(folder, base + ".json"), json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
            return dict(meta, file=base + ".md")

    # ---- vectors for the novelty check -----------------------------------------------------------------
    def save_vector(self, resident: str, work_file: str, vector: List[float], model: str = "") -> None:
        folder = os.path.join(self.root(resident), "vectors")
        os.makedirs(folder, exist_ok=True)
        base = work_file[:-3] if work_file.endswith(".md") else work_file
        _write(os.path.join(folder, base + ".json"),
               json.dumps({"model": model, "vector": [round(float(x), 6) for x in vector]}) + "\n")

    def load_vectors(self, resident: str) -> Dict[str, List[float]]:
        """work file (NNNN-slug.md) -> its vector, for the works that have one."""
        folder = os.path.join(self.root(resident), "vectors")
        out: Dict[str, List[float]] = {}
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            if name.endswith(".json"):
                try:
                    with open(os.path.join(folder, name), encoding="utf-8") as f:
                        out[name[:-5] + ".md"] = [float(x) for x in json.load(f)["vector"]]
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        return out

    def save_draft(self, resident: str, session: str, text: str, note: str = "") -> str:
        folder = os.path.join(self.root(resident), "drafts")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, session + ".md")
        _write(path, (text.rstrip() + "\n" if text else "") + ("\n[" + note + "]\n" if note else ""))
        return path

    def drop_draft(self, resident: str, session: str) -> None:
        try:
            os.remove(os.path.join(self.root(resident), "drafts", session + ".md"))
        except OSError:
            pass

    def append_journal(self, resident: str, session: str, text: str) -> None:
        with self.lock:
            with open(os.path.join(self.root(resident), "journal.md"), "a", encoding="utf-8", newline="\n") as f:
                f.write("<!-- entry ts=%.3f session=%s -->\n%s\n\n" % (self.clock(), session, text.strip()))

    def read_journal(self, resident: str) -> str:
        with open(os.path.join(self.root(resident), "journal.md"), encoding="utf-8") as f:
            return f.read()

    def journal_entries(self, resident: str) -> List[Dict[str, Any]]:
        text = self.read_journal(resident)
        marks = list(_JOURNAL_MARK.finditer(text))
        out = []
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            out.append({"ts": float(m.group(1)), "session": m.group(2), "text": text[m.end():end].strip()})
        return out

    def add_wish(self, resident: str, session: str, text: str) -> None:
        with self.lock:
            with open(os.path.join(self.root(resident), "wishes.jsonl"), "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps({"ts": round(self.clock(), 3), "session": session, "text": text.strip()}, ensure_ascii=False) + "\n")

    def _wish_lines(self, resident: str) -> List[Dict[str, Any]]:
        """Every line of wishes.jsonl: wishes, model requests, and the decision / receipt lines that are appended about requests."""
        out = []
        try:
            with open(os.path.join(self.root(resident), "wishes.jsonl"), encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict):
                        out.append(rec)
        except OSError:
            pass
        return out

    def wishes(self, resident: str) -> List[Dict[str, Any]]:
        """Wishes and model requests (a request carries its folded status); decision and receipt lines are not wishes."""
        folded = {r["id"]: r for r in self.model_requests(resident)}
        return [dict(folded[r["id"]]) if r.get("kind") == "model_request" and r.get("id") in folded else r
                for r in self._wish_lines(resident) if r.get("kind") in WISH_KINDS]

    # ---- model requests (G4): appended to wishes.jsonl, never rewritten ---------------------------------
    def model_requests(self, resident: str = "", status: str = "") -> List[Dict[str, Any]]:
        """The model requests of one resident (or of every resident), oldest first, each with its current status. A decision is a
        line appended after the request (`model_decision`), so the file stays append-only and the history stays readable."""
        out: List[Dict[str, Any]] = []
        for rid in ([resident] if resident else self.residents()):
            if not self.exists(rid):
                continue
            by_id: Dict[str, Dict[str, Any]] = {}
            for rec in self._wish_lines(rid):
                kind = rec.get("kind")
                if kind == "model_request" and rec.get("id"):
                    by_id[rec["id"]] = dict(rec, resident=rid, status="pending")
                elif kind == "model_decision" and rec.get("request") in by_id:
                    r = by_id[rec["request"]]
                    if r["status"] == "pending":
                        r.update(status=rec.get("status"), decided_by=rec.get("by"), decided_at=rec.get("ts"), note=rec.get("note", ""))
                elif kind == "model_outcome_seen" and rec.get("request") in by_id:
                    by_id[rec["request"]]["reported_in"] = rec.get("session")
            out += [r for r in by_id.values() if not status or r["status"] == status]
        return out

    def add_model_request(self, resident: str, session: str, model: str, why: str) -> Dict[str, Any]:
        with self.lock:
            n = len([r for r in self._wish_lines(resident) if r.get("kind") == "model_request"]) + 1
            rec = {"ts": round(self.clock(), 3), "session": session, "kind": "model_request", "id": "%s.r%05d" % (resident, n),
                   "model": model, "why": why, "text": "Asked to run on %s. Reason: %s" % (model, why), "status": "pending"}
            with open(os.path.join(self.root(resident), "wishes.jsonl"), "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec

    def decide_request(self, request_id: str, status: str, by: str = "owner", note: str = "") -> Dict[str, Any]:
        """granted or declined, once. Appends a decision line; a request already decided is refused."""
        m = REQUEST_ID.match(str(request_id or ""))
        if not m or not self.exists(m.group(1)):
            raise StoreError("no such request")
        if status not in ("granted", "declined"):
            raise StoreError("status must be granted or declined")
        rid = m.group(1)
        with self.lock:
            req = next((r for r in self.model_requests(rid) if r["id"] == request_id), None)
            if req is None:
                raise StoreError("no such request")
            if req["status"] != "pending":
                raise StoreError("that request was already %s" % req["status"])
            rec = {"ts": round(self.clock(), 3), "kind": "model_decision", "request": request_id, "status": status,
                   "by": " ".join(str(by or "owner").split())[:60] or "owner", "note": " ".join(str(note or "").split())[:500]}
            with open(os.path.join(self.root(rid), "wishes.jsonl"), "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return dict(req, status=status, decided_by=rec["by"], decided_at=rec["ts"], note=rec["note"])

    def unreported_outcomes(self, resident: str) -> List[Dict[str, Any]]:
        """Decided requests the resident has not been told about yet."""
        return [r for r in self.model_requests(resident) if r["status"] != "pending" and not r.get("reported_in")]

    def mark_outcomes_reported(self, resident: str, session: str, ids: Iterable[str]) -> None:
        with self.lock:
            with open(os.path.join(self.root(resident), "wishes.jsonl"), "a", encoding="utf-8", newline="\n") as f:
                for i in ids:
                    f.write(json.dumps({"ts": round(self.clock(), 3), "kind": "model_outcome_seen", "request": i, "session": session}) + "\n")

    def catalogue(self) -> Optional[Dict[str, Any]]:
        """The model catalogue StoneSage pushed ({configured: [...], other: [...]}), or None when none has arrived."""
        snap = self.read_snapshot(MODELS_KIND)
        if not snap:
            return None
        try:
            cat = json.loads(snap.get("text") or "")
        except ValueError:
            return None
        if not isinstance(cat, dict):
            return None
        return {"configured": [m for m in cat.get("configured") or [] if isinstance(m, dict) and m.get("name")],
                "other": [m for m in cat.get("other") or [] if isinstance(m, dict) and m.get("name")], "at": snap.get("at")}

    # ---- session transcripts (append-only) -----------------------------------------------------
    def transcript_path(self, resident: str, session: str) -> str:
        if not SESSION_ID.match(session or ""):
            raise StoreError("bad session id")
        return os.path.join(self.root(resident), "sessions", session + ".jsonl")

    def append_record(self, resident: str, session: str, record: Dict[str, Any]) -> None:
        path = self.transcript_path(resident, session)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        record = dict(record, ts=round(self.clock(), 3))
        with self.lock:
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_transcript(self, resident: str, session: str) -> List[Dict[str, Any]]:
        out = []
        try:
            with open(self.transcript_path(resident, session), encoding="utf-8") as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        return out

    def find_session(self, session: str) -> Optional[str]:
        """The resident whose folder holds this session's transcript."""
        if not SESSION_ID.match(session or ""):
            return None
        souls = os.path.join(self.data_dir, "souls")
        for resident in sorted(os.listdir(souls)):
            if os.path.isfile(os.path.join(souls, resident, "sessions", session + ".jsonl")):
                return resident
        return None

    def session_summary(self, resident: str, session: str) -> Dict[str, Any]:
        recs = self.read_transcript(resident, session)
        head = next((r for r in recs if r.get("type") == "header"), {})
        end = next((r for r in reversed(recs) if r.get("type") == "end"), None)
        web = [{"action": r.get("action"), "target": r.get("query") or r.get("url") or "", "domain": r.get("domain", ""),
                "bytes": r.get("bytes", 0), "outcome": r.get("outcome", "")} for r in recs if r.get("type") == "web"]
        reads = [{"post": r.get("post", ""), "count": r.get("count", 0)} for r in recs if r.get("type") == "comments_read"]
        windows = [r.get("kind", "") for r in recs if r.get("type") == "window"]
        cross = [{"action": r.get("action") or r["type"][len("cross_"):], "post": r.get("post", ""), "count": r.get("count")}
                 for r in recs if r.get("type") in ("cross_read", "cross_comment", "cross_rating")][:MAX_WEB_LOG]
        asked = [{"id": r.get("id", ""), "model": r.get("model", "")} for r in recs if r.get("type") == "model_request"]
        return {"session": session, "resident": resident, "started": head.get("ts"),
                "model": (head.get("model") or {}).get("name"), "state": "ended" if end else "running",
                "reason": (end or {}).get("reason"), "turns": (end or {}).get("turns"),
                "tokens": (end or {}).get("tokens"), "ended": (end or {}).get("ts"), "records": len(recs),
                "web": web[:MAX_WEB_LOG], "web_total": len(web), "comments_read": reads, "window": windows,
                "cross": cross, "model_requests": asked}

    # ---- the board -----------------------------------------------------------------------------
    def residents(self) -> List[str]:
        souls = os.path.join(self.data_dir, "souls")
        return sorted(r for r in os.listdir(souls) if AGENT_ID.match(r) and self.exists(r))

    @staticmethod
    def post_id(resident: str, kind: str, key: Any) -> str:
        return "%s:%s:%s" % (resident, kind, key)

    def board(self, since: float = 0.0, resident: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        """Posts, newest first: works, journal entries, drafts, wishes and session summaries. Each has an `id` (resident:kind:key)
        and the count / average of the comments and ratings it has received."""
        posts: List[Dict[str, Any]] = []
        for rid in ([resident] if resident else self.residents()):
            if not self.exists(rid):
                continue
            name = self.manifest(rid).get("name", rid)
            stats = self.comment_stats(rid)
            mine: List[Dict[str, Any]] = []
            for meta in self.list_works(rid):
                extra = {k: meta[k] for k in ("novelty", "variation_on", "summary", "cross_read") if k in meta}
                mine.append({"kind": "work", "key": meta["file"][:-3], "ts": meta.get("at", 0), "title": meta.get("title"),
                             "form": meta.get("form"), "session": meta.get("session"), "model": meta.get("model"),
                             "text": self.read_work_text(rid, meta)[:8000], **extra})
            for e in self.journal_entries(rid):
                mine.append({"kind": "journal", "key": "%.3f" % e["ts"], "ts": e["ts"], "session": e["session"], "text": e["text"][:8000]})
            for w in self.wishes(rid):
                post = {"kind": "wish", "key": "%.3f" % w.get("ts", 0), "ts": w.get("ts", 0), "session": w.get("session"),
                        "text": str(w.get("text", ""))[:2000]}
                if w.get("kind") == "model_request":
                    post["request"] = {k: w.get(k) for k in ("id", "model", "why", "status", "decided_by", "note") if w.get(k) is not None}
                mine.append(post)
            drafts = os.path.join(self.root(rid), "drafts")
            for fn in sorted(os.listdir(drafts)) if os.path.isdir(drafts) else []:
                if fn.endswith(".md"):
                    full = os.path.join(drafts, fn)
                    with open(full, encoding="utf-8") as f:
                        mine.append({"kind": "draft", "key": fn[:-3], "ts": round(os.path.getmtime(full), 3), "session": fn[:-3],
                                     "text": f.read()[:8000]})
            sdir = os.path.join(self.root(rid), "sessions")
            for fn in sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []:
                if fn.endswith(".jsonl"):
                    s = self.session_summary(rid, fn[:-6])
                    mine.append({"kind": "session", "key": s["session"], "session": s["session"],
                                 "ts": s["ended"] or s["started"] or 0, "model": s["model"], "state": s["state"],
                                 "reason": s["reason"], "turns": s["turns"], "tokens": s["tokens"], "web": s["web"],
                                 "web_total": s["web_total"], "comments_read": s["comments_read"], "window": s["window"],
                                 "cross": s["cross"], "model_requests": s["model_requests"]})
            for p in mine:
                pid = self.post_id(rid, p["kind"], p.pop("key"))
                st = stats.get(pid, {})
                p.update({"id": pid, "resident": rid, "name": name, "comments": st.get("comments", 0),
                          "rating": st.get("rating"), "ratings": st.get("ratings", 0),
                          "resident_comments": st.get("resident_comments", 0), "resident_rating": st.get("resident_rating"),
                          "resident_ratings": st.get("resident_ratings", 0)})
            posts += mine
        posts = [p for p in posts if (p.get("ts") or 0) > since]
        posts.sort(key=lambda p: p.get("ts") or 0, reverse=True)
        return posts[:max(1, min(int(limit), 1000))]

    # ---- comments and ratings -------------------------------------------------------------------------
    def post_exists(self, post_id: str) -> bool:
        m = POST_ID.match(str(post_id or ""))
        if not m or not self.exists(m.group(1)):
            return False
        rid, kind, key = m.groups()
        root = self.root(rid)
        if kind == "work":
            return os.path.isfile(os.path.join(root, "works", key + ".json"))
        if kind == "draft":
            return os.path.isfile(os.path.join(root, "drafts", key + ".md"))
        if kind == "session":
            return SESSION_ID.match(key) is not None and os.path.isfile(os.path.join(root, "sessions", key + ".jsonl"))
        if kind == "journal":
            return any("%.3f" % e["ts"] == key for e in self.journal_entries(rid))
        return any("%.3f" % w.get("ts", 0) == key for w in self.wishes(rid))

    def add_comment(self, post_id: str, author: str, text: str = "", rating: Any = None, kind: str = "user",
                    author_id: str = "") -> Dict[str, Any]:
        """A comment, a 1-5 rating, or both, by `author` on a post. Appended to that post's resident's comments.jsonl.
        `kind` is "user" (a person, through StoneSage) or "agent" (a resident; `author` is its chosen name, `author_id` its id)."""
        m = POST_ID.match(str(post_id or ""))
        if not m or not self.post_exists(post_id):
            raise StoreError("no such post")
        if kind not in ("user", "agent"):
            raise StoreError("a comment is by a user or an agent")
        limit = MAX_AGENT_COMMENT_CHARS if kind == "agent" else MAX_COMMENT_CHARS
        text = str(text or "").strip()[:limit]
        if rating is not None and (isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5):
            raise StoreError("a rating is a whole number from 1 to 5")
        if not text and rating is None:
            raise StoreError("a comment needs text or a rating")
        author = " ".join(str(author or "user").split())[:60] or "user"
        rid = m.group(1)
        with self.lock:
            n = len(self._comment_lines(rid)) + 1
            rec = {"id": "c%05d" % n, "post": post_id, "ts": round(self.clock(), 3), "author": author, "text": text, "rating": rating}
            if kind == "agent":
                rec.update(kind="agent", resident=str(author_id or ""))
            with open(os.path.join(self.root(rid), "comments.jsonl"), "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def _comment_lines(self, resident: str) -> List[Dict[str, Any]]:
        out = []
        try:
            with open(os.path.join(self.root(resident), "comments.jsonl"), encoding="utf-8") as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        return out

    def comments(self, resident: str, post_id: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        """Newest first; one post's, or (no post_id) all of this resident's."""
        rows = [c for c in self._comment_lines(resident) if not post_id or c.get("post") == post_id]
        rows.sort(key=lambda c: (c.get("ts", 0), c.get("id", "")), reverse=True)
        return rows[:max(1, min(int(limit), 500))]

    def comment_stats(self, resident: str) -> Dict[str, Dict[str, Any]]:
        """post id -> {comments, rating, ratings, resident_comments, resident_rating, resident_ratings}. `comments` counts everyone's;
        `rating` is the average of each USER's latest rating and `resident_rating` the same for the residents (one per resident id),
        so the two groups are never mixed."""
        stats: Dict[str, Dict[str, Any]] = {}
        latest: Dict[str, Dict[str, Dict[str, int]]] = {}
        for c in self._comment_lines(resident):
            pid = c.get("post", "")
            st = stats.setdefault(pid, {"comments": 0, "rating": None, "ratings": 0})
            agent = c.get("kind") == "agent"
            if c.get("text"):
                st["comments"] += 1
                if agent:                          # the resident_* figures exist only for posts residents have touched
                    st["resident_comments"] = st.get("resident_comments", 0) + 1
            if isinstance(c.get("rating"), int):
                who = str(c.get("resident") or c.get("author")) if agent else str(c.get("author"))
                latest.setdefault(pid, {}).setdefault("agent" if agent else "user", {})[who] = c["rating"]
        for pid, groups in latest.items():
            for group, prefix in (("user", ""), ("agent", "resident_")):
                vals = list((groups.get(group) or {}).values())
                if vals:
                    stats[pid][prefix + "ratings"] = len(vals)
                    stats[pid][prefix + "rating"] = round(sum(vals) / len(vals), 2)
        return stats

    # ---- other residents' finished works (G5) ---------------------------------------------------------
    def display_name(self, resident: str) -> str:
        try:
            return str(self.manifest(resident).get("name") or resident)
        except (OSError, ValueError):
            return resident

    def cross_works(self, viewer: str, hidden: Iterable[str] = (), only: str = "") -> List[Dict[str, Any]]:
        """Finished works by residents other than `viewer`, newest first, each {post, resident, name, meta}. `hidden` residents (those who
        opted out of cross-reading) are left out. `only` narrows to one resident by id or chosen name (case-insensitive). Works only:
        never journals, drafts, wishes or sessions."""
        skip = set(hidden)
        want = " ".join(str(only or "").split()).lower()
        out: List[Dict[str, Any]] = []
        for rid in self.residents():
            if rid == viewer or rid in skip:
                continue
            name = self.display_name(rid)
            if want and want not in (rid.lower(), name.lower()):
                continue
            for meta in self.list_works(rid):
                out.append({"post": self.post_id(rid, "work", meta["file"][:-3]), "resident": rid, "name": name, "meta": meta})
        out.sort(key=lambda w: w["meta"].get("at", 0), reverse=True)
        return out

    def cross_work(self, viewer: str, post_id: str, hidden: Iterable[str] = ()) -> Dict[str, Any]:
        """One finished work by another visible resident, else StoreError (the message never says whether a hidden one exists)."""
        m = POST_ID.match(str(post_id or ""))
        if not m or m.group(2) != "work":
            raise StoreError("that is not the id of a finished work (read_board lists them)")
        rid = m.group(1)
        if rid == viewer:
            raise StoreError("that is one of your own works")
        if rid in set(hidden) or not self.post_exists(post_id):
            raise StoreError("no such finished work")
        meta = next((w for w in self.list_works(rid) if w["file"][:-3] == m.group(3)), None)
        if meta is None:
            raise StoreError("no such finished work")
        return {"post": post_id, "resident": rid, "name": self.display_name(rid), "meta": meta}

    def last_session_start(self, resident: str, before: str = "") -> float:
        """When this resident's previous session began (the header's time), or 0.0 when there was none. `before` = the current session."""
        folder = os.path.join(self.root(resident), "sessions")
        names = sorted(f[:-6] for f in os.listdir(folder) if f.endswith(".jsonl")) if os.path.isdir(folder) else []
        for sid in reversed([n for n in names if n != before]):
            head = next((r for r in self.read_transcript(resident, sid) if r.get("type") == "header"), None)
            if head and head.get("ts"):
                return float(head["ts"])
        return 0.0

    # ---- window snapshots ---------------------------------------------------------------------------
    def save_snapshot(self, kind: str, text: str) -> Dict[str, Any]:
        if not KIND.match(str(kind or "")):
            raise StoreError("a snapshot kind is 1-40 characters of a-z, 0-9, '_' and '-'")
        text = str(text or "")
        if len(text) > MAX_SNAPSHOT_CHARS:
            raise StoreError("snapshot too large")
        rec = {"kind": kind, "at": round(self.clock(), 3), "text": text}
        with self.lock:
            _write(os.path.join(self.data_dir, "snapshots", kind + ".json"), json.dumps(rec, ensure_ascii=False) + "\n")
        return {"kind": kind, "at": rec["at"], "chars": len(text)}

    def read_snapshot(self, kind: str) -> Optional[Dict[str, Any]]:
        if not KIND.match(str(kind or "")):
            return None
        try:
            with open(os.path.join(self.data_dir, "snapshots", kind + ".json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def household_terms(self) -> List[str]:
        snap = self.read_snapshot(TERMS_KIND)
        return [t for t in (snap or {}).get("text", "").splitlines() if t.strip()]
