"""One garden session: the invitation, the resident's tools, the budgets, the degeneration breaker, the raw transcript.

The transcript (`sessions/<id>.jsonl`, append-only) holds a header with everything needed to repeat the run, then one record per
message sent to the model, per answer received, per tool call and result, and a closing record with the reason it ended.
"""

import json
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from . import egress, embed, texts, web as webmod
from .store import MAX_AGENT_COMMENT_CHARS, MAX_PENDING_REQUESTS, WINDOW_KINDS, Store, StoreError

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty", "frequency_penalty")
MAX_EMPTY_REPLIES = 2           # replies with neither text nor a tool call, in a row, before the session ends
MAX_JOURNAL_CHARS = 20000       # a plain-text reply kept in the journal is cut to this
RESULT_CHARS = 8000             # the longest tool result handed back to the model
BAD_ARGS_NOTE = ("your arguments were cut off or malformed (a reply that runs past the per-call length limit ends mid-argument). "
                 "Long text can be written in parts: call write_draft with what you have, add to it, then call finish_work.")
MAX_OPENING_WORKS = 40          # the opening digest lists this many works, newest first
OPENING_JOURNAL = 5             # ... and the first line of this many of the newest journal entries
DIGEST_CHARS = 120              # the first line of a work or journal entry, as the opening shows it
SUMMARY_CHARS = 300             # a summary the resident writes with finish_work
ROOM_SHARE = 0.85               # the room note is sent when the last prompt passes this share of GARDEN_MAX_CONTEXT_TOKENS
MAX_JOURNAL_LAST = 200          # read_journal(last=N) returns at most this many entries
MAX_AGENT_COMMENTS = 10        # comments a resident may leave on others' works in one session
CROSS_LIST_DEFAULT, CROSS_LIST_MAX = 20, 50
CHARS_PER_TOKEN_ESTIMATE = 3    # only when the engine returns no prompt_tokens: a deliberately high estimate


def first_line(text: str, limit: int = DIGEST_CHARS) -> str:
    """The first non-empty line of `text`, cut to `limit` characters."""
    line = next((ln.strip() for ln in (text or "").splitlines() if ln.strip()), "")
    return line if len(line) <= limit else line[:limit - 3].rstrip() + "..."


def day(at: Any) -> str:
    """A stored timestamp as a UTC date (YYYY-MM-DD)."""
    try:
        return time.strftime("%Y-%m-%d", time.gmtime(float(at or 0)))
    except (TypeError, ValueError, OverflowError, OSError):
        return "unknown date"


def work_digest(meta: Dict[str, Any], text: str) -> str:
    """What the opening shows for one work: the summary the resident wrote with finish_work, or, for a work without one, a
    deterministic fallback: its word count and the first line. No model is asked."""
    summary = " ".join(str(meta.get("summary") or "").split())
    if summary:
        return summary[:SUMMARY_CHARS]
    words = meta.get("words")
    if not isinstance(words, int):
        words = len((text or "").split())
    return "%d words, begins: %s" % (words, first_line(text))


# ---- the degeneration breaker ------------------------------------------------------------------------------
def degeneration(text: str, sentence_repeats: int = 3, ngram: int = 6, ngram_repeats: int = 5, word_run: int = 20) -> Optional[str]:
    """A reason when `text` has fallen into a loop, else None. Deterministic: repeated sentences, one repeated word-sequence,
    or one word over and over. No model is asked."""
    if not text:
        return None
    sentences: Dict[str, int] = {}
    for s in re.split(r"[.!?\n]+", text):
        norm = " ".join(s.lower().split())
        if len(norm) >= 25:
            sentences[norm] = sentences.get(norm, 0) + 1
    for norm, n in sentences.items():
        if n >= sentence_repeats:
            return "repeated sentence x%d: %s" % (n, norm[:60])
    words = re.findall(r"\w+", text.lower())
    grams: Dict[tuple, int] = {}
    for i in range(len(words) - ngram + 1):
        g = tuple(words[i:i + ngram])
        grams[g] = grams.get(g, 0) + 1
    for g, n in grams.items():
        if n >= ngram_repeats and len(set(g)) > 1:
            return "repeated phrase x%d: %s" % (n, " ".join(g)[:60])
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        if run >= word_run:
            return "one word repeated %d times: %s" % (run, a[:30])
    return None


def parse_args(raw: Any) -> Dict[str, Any]:
    """A tool call's arguments as an object. Raises ValueError when they are not one (a string that is not JSON, or not an object)."""
    args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw if isinstance(raw, dict) else {})
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    return args


def replayable_call(call: Dict[str, Any]) -> Dict[str, Any]:
    """The copy of a model's tool call that goes back into the conversation history. Arguments that do not parse are replaced
    with "{}" (same id and name, so the tool error that follows still pairs with it): the chat template of the engine fails on
    broken JSON in history and would end the whole session. The raw transcript keeps the original, untouched."""
    fn = dict(call.get("function") or {})
    raw = fn.get("arguments")
    try:
        parse_args(raw)
    except ValueError:
        fn["arguments"] = "{}"
    else:
        if not isinstance(raw, str):
            fn["arguments"] = json.dumps(raw if isinstance(raw, dict) else {}, ensure_ascii=False)
    return {**call, "function": fn}


# ---- the model --------------------------------------------------------------------------------------------
class ModelError(RuntimeError):
    pass


class ToolParseError(ModelError):
    """The engine refused to parse the model's own tool-call JSON (its grammar choked on it, e.g. a raw newline in a string).
    The session retries that request once without engine-side parsing and reads the tool calls out of the text."""


PARSE_FAILURE = "Failed to parse tool call"
TEXT_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


class HttpModel:
    """POSTs a chat body to StoneSage's garden proxy (or any OpenAI-style endpoint) with the garden model key."""

    def __init__(self, url: str, key: str, timeout: float = 600.0):
        url = url.rstrip("/")
        self.url = url if url.endswith("/chat/completions") else url + "/chat/completions"
        self._key = key
        self.timeout = timeout

    def __call__(self, body: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._key})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read(4096).decode("utf-8", "replace")
            except OSError:
                detail = ""
            if PARSE_FAILURE in detail:
                raise ToolParseError("model HTTP %d: %s" % (e.code, PARSE_FAILURE)) from None
            raise ModelError("model HTTP %d" % e.code) from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise ModelError(type(e).__name__) from None


def parse_text_calls(text: str, id_prefix: str = "text") -> Tuple[List[Dict[str, Any]], str]:
    """Tool calls written in the reply's text, for a model whose engine would not parse them. Qwen style
    `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks; failing that, a whole reply that is one JSON object with a
    name. Raw newlines and control characters inside strings are accepted (strict=False). -> (tool_calls in the chat shape, the
    text without the blocks). Nothing parsed -> ([], text) unchanged: the caller keeps it as plain text."""
    blocks = [m.group(1) for m in TEXT_CALL.finditer(text or "")]
    if not blocks and (text or "").strip().startswith("{"):
        blocks = [text.strip()]
    calls: List[Dict[str, Any]] = []
    for raw in blocks:
        try:
            obj = json.loads(raw, strict=False)
        except ValueError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"]:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args, strict=False)
            except ValueError:
                args = None
        args_text = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=False)
        calls.append({"id": "%s-%d" % (id_prefix, len(calls) + 1), "type": "function",
                      "function": {"name": obj["name"], "arguments": args_text}})
    if not calls:
        return [], text
    rest = TEXT_CALL.sub("", text).strip() if TEXT_CALL.search(text) else ""
    return calls, rest


def new_session_id(now: Optional[float] = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if now is None else now)) + "-" + secrets.token_hex(2)


# ---- the session ----------------------------------------------------------------------------------------
class Session:
    def __init__(self, store: Store, resident: str, model: Callable[[Dict[str, Any]], Dict[str, Any]], model_name: str,
                 model_sha256: str, sampling: Optional[Dict[str, Any]] = None, seed: Optional[int] = None,
                 max_tokens: int = 16000, max_turns: int = 40, max_call_tokens: int = 4096, session_id: str = "",
                 web: Optional[Any] = None, embedder: Optional[Callable[[List[str]], List[List[float]]]] = None,
                 novelty_threshold: float = embed.DEFAULT_THRESHOLD, max_web_calls: int = 20,
                 max_context_tokens: int = 28000, duplicate_threshold: float = embed.DEFAULT_DUPLICATE_THRESHOLD,
                 max_seconds: Optional[float] = None, cross_reading: bool = True, hidden_residents: Optional[Iterable[str]] = None):
        self.store, self.resident, self.model = store, resident, model
        self.cross_reading = bool(cross_reading)                 # may this resident read, comment on and rate others' finished works?
        self.hidden = {str(h) for h in (hidden_residents or ())}   # residents who opted out: their works are not shown to anyone
        self._cross_read: List[str] = []                         # post ids of other residents' works read (in full) this session
        self._agent_comments = 0
        self.duplicate_threshold = float(duplicate_threshold)
        self.max_seconds = float(max_seconds) if max_seconds else None      # a wall-clock deadline (the scheduler's window); None = none
        self._deadline_timer: Optional[threading.Timer] = None
        self.web, self.embedder = web, embedder                 # the fetcher client (search / read) and the embeddings call
        self.novelty_threshold, self.max_web_calls, self.web_calls = float(novelty_threshold), int(max_web_calls), 0
        self.model_name, self.model_sha256 = str(model_name or ""), str(model_sha256 or "")
        self.sampling = {k: v for k, v in (sampling or {}).items() if k in SAMPLING_KEYS and isinstance(v, (int, float))}
        self.seed = int(seed) if seed is not None else secrets.randbelow(2 ** 31)
        self.max_tokens, self.max_turns, self.max_call_tokens = int(max_tokens), int(max_turns), int(max_call_tokens)
        self.max_context = int(max_context_tokens)
        self.id = session_id or new_session_id(store.clock())
        self.messages: List[Dict[str, Any]] = []
        self.draft = ""
        # The budget: `tokens` is the session's GENERATED (completion) tokens, the sum over every reply. Each call asks for at most
        # what is left of max_tokens. The conversation's size is separate: `prompt_tokens` is the last call's prompt (what the
        # engine counted), checked against max_context before each call.
        self.tokens = 0
        self.prompt_tokens = 0
        self._sent_upto = 0                                     # messages[:_sent_upto] were in the last prompt
        self._room_noted = False
        self.turns = 0
        self.state = "new"
        self.reason = ""
        self.thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._stop_reason = ""
        self._finished_flag = False
        self._empty = 0

    # ---- lifecycle --------------------------------------------------------------------------------
    def start(self) -> None:
        """Write the header, then run the loop on a thread."""
        self.store.create_resident(self.resident)
        self.state = "running"
        self._record({"type": "header", "session": self.id, "resident": self.resident,
                      "invitation": {"version": texts.INVITATION_VERSION, "text": texts.INVITATION},
                      "opening": {"version": texts.OPENING_VERSION, "max_works": MAX_OPENING_WORKS, "journal_entries": OPENING_JOURNAL,
                                  "digest_chars": DIGEST_CHARS, "summary_chars": SUMMARY_CHARS},
                      "room_note": {"version": texts.ROOM_NOTE_VERSION, "text": texts.ROOM_NOTE, "share": ROOM_SHARE},
                      "persona": {"version": self.store.manifest(self.resident).get("persona_version", ""),
                                  "text": self.store.persona(self.resident)},
                      "toolset": {"version": texts.TOOLSET_VERSION, "sha256": texts.toolset_sha256(),
                                  "names": [t["function"]["name"] for t in texts.TOOLS],
                                  "exposed": [t["function"]["name"] for t in self._tools()],
                                  "exposed_sha256": texts.toolset_sha256(self._tools())},
                      "cross_reading": {"on": self.cross_reading, "hidden_residents": sorted(self.hidden),
                                        "max_comments_per_session": MAX_AGENT_COMMENTS},
                      "model": {"name": self.model_name, "sha256": self.model_sha256},
                      "sampling": self.sampling, "seed": self.seed,
                      "budgets": {"max_tokens": self.max_tokens, "max_turns": self.max_turns,
                                  "max_call_tokens": self.max_call_tokens, "max_web_calls": self.max_web_calls,
                                  "max_context_tokens": self.max_context,
                                  "counts": "tokens = generated (completion) tokens, summed over replies; context = the last prompt"},
                      "novelty_threshold": self.novelty_threshold, "duplicate_threshold": self.duplicate_threshold,
                      "max_seconds": self.max_seconds, "web": bool(self.web)})
        if self.max_seconds:
            self._deadline_timer = threading.Timer(self.max_seconds, self.request_stop, args=("deadline",))
            self._deadline_timer.daemon = True
            self._deadline_timer.start()
        self.thread = threading.Thread(target=self._run, name="garden-" + self.id, daemon=True)
        self.thread.start()

    def request_stop(self, reason: str = "") -> None:
        """Preemption: Computer needs the engine. The loop ends within a moment, the draft is kept."""
        self._stop_reason = (reason or "stop requested")[:200]
        self._stop.set()

    def wait(self, timeout: float = 10.0) -> bool:
        if self.thread:
            self.thread.join(timeout)
        return self.state == "ended"

    def status(self) -> Dict[str, Any]:
        return {"session": self.id, "resident": self.resident, "state": self.state, "reason": self.reason,
                "turns": self.turns, "tokens": self.tokens, "model": self.model_name}

    def _record(self, rec: Dict[str, Any]) -> None:
        self.store.append_record(self.resident, self.id, rec)

    def _send(self, message: Dict[str, Any]) -> None:
        self.messages.append(message)
        self._record({"type": "sent", "message": message})

    # ---- the loop ------------------------------------------------------------------------------------
    def _work_text(self, meta: Dict[str, Any]) -> str:
        try:
            return self.store.read_work_text(self.resident, meta)
        except (OSError, KeyError):
            return ""

    def _opening(self) -> None:
        """The first user message: a digest of the resident's own past, not the texts. Works newest first (their summaries, or a
        fallback digest), and the journal's size with the first line of its newest entries. The full texts stay one tool call away
        (read_work, read_journal). Neutral: nothing here suggests what to do."""
        m = self.store.manifest(self.resident)
        works = self.store.list_works(self.resident)
        entries = self.store.journal_entries(self.resident)
        lines = ["Your name is %s." % m.get("name") if m.get("named") else "You have no name yet."]
        if entries:
            lines.append("Your journal has %d entries. The newest %d, first line of each, newest first:"
                         % (len(entries), min(OPENING_JOURNAL, len(entries))))
            lines += ["- %s: %s" % (day(e.get("ts")), first_line(e.get("text", ""))) for e in reversed(entries[-OPENING_JOURNAL:])]
        else:
            lines.append("Your journal is empty.")
        if works:
            shown = list(reversed(works))[:MAX_OPENING_WORKS]
            head = "Your works, newest first"
            if len(works) > len(shown):
                head += " (the newest %d of %d)" % (len(shown), len(works))
            lines.append(head + ":")
            for w in shown:
                body = work_digest(w, "" if w.get("summary") else self._work_text(w))
                lines.append("- %s (%s, %s): %s" % (w.get("title"), w.get("form"), day(w.get("at")), body))
        else:
            lines.append("You have no finished works yet.")
        lines += self._request_lines()
        lines += self._garden_lines()
        lines.append("read_work(title) returns a work in full; read_journal() returns the whole journal, read_journal(last=N) the newest N entries.")
        self._send({"role": "system", "content": self.store.persona(self.resident) + "\n\n" + texts.INVITATION})
        self._send({"role": "user", "content": "\n".join(lines)})

    def _request_lines(self) -> List[str]:
        """How earlier model requests came out, once each: decided and not yet reported. Pending ones are not mentioned."""
        decided = self.store.unreported_outcomes(self.resident)
        if not decided:
            return []
        lines = []
        for r in decided:
            note = r.get("note") or ""
            if r["status"] == "granted":
                line = "Your request for %s: granted." % r.get("model")
                if r.get("model") == self.model_name:
                    line += " This session runs on it."
            else:
                line = "Your request for %s: declined%s" % (r.get("model"), ": " + note if note else ".")
            lines.append(line)
        self.store.mark_outcomes_reported(self.resident, self.id, [r["id"] for r in decided])
        self._record({"type": "requests_reported", "requests": [{"id": r["id"], "model": r.get("model"), "status": r["status"]} for r in decided]})
        return lines

    def _garden_lines(self) -> List[str]:
        """One line when other residents exist: that they do, and how many new finished works they have since this resident's last
        session. No titles, names or content: reading them stays a choice."""
        if not self.cross_reading:
            return []
        since = self.store.last_session_start(self.resident, self.id)
        works = self.store.cross_works(self.resident, self.hidden)
        others = {w["resident"] for w in works}
        for rid in self.store.residents():
            if rid != self.resident and rid not in self.hidden:
                others.add(rid)
        if not others:
            return []
        new = sum(1 for w in works if (w["meta"].get("at") or 0) > since)
        self._record({"type": "garden_digest", "other_residents": len(others), "new_finished_works": new, "since": since})
        return ["There %s %d other resident%s in this garden, with %d new finished work%s since your last session. "
                "read_board() lists them; reading them is optional."
                % ("is" if len(others) == 1 else "are", len(others), "" if len(others) == 1 else "s", new, "" if new == 1 else "s")]

    def _tools(self) -> List[Dict[str, Any]]:
        """The tool schemas for this call: name_self is left out once the resident has a name (it could never succeed), and the
        cross-reading tools while cross-reading is off for this resident."""
        skip = set()
        if self.store.manifest(self.resident).get("named"):
            skip.add("name_self")
        if not self.cross_reading:
            skip.update(texts.CROSS_TOOLS)
        return [t for t in texts.TOOLS if t["function"]["name"] not in skip] if skip else texts.TOOLS

    def _projected_prompt(self) -> int:
        """The prompt the next call would send: the last call's prompt_tokens plus the messages added since, at a high character
        estimate. Before the first call, the whole conversation so far."""
        added = sum(len(json.dumps(m, ensure_ascii=False)) for m in self.messages[self._sent_upto:])
        return self.prompt_tokens + added // CHARS_PER_TOKEN_ESTIMATE

    def _call(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The model call on a helper thread, so a stop request ends the wait at once (the abandoned call times out by itself)."""
        box: Dict[str, Any] = {}
        done = threading.Event()

        def work() -> None:
            try:
                box["resp"] = self.model(body)
            except Exception as e:
                box["exc"] = e
                box["err"] = "%s: %s" % (type(e).__name__, e)
            done.set()

        threading.Thread(target=work, daemon=True).start()
        while not done.wait(0.1):
            if self._stop.is_set():
                return None
        if isinstance(box.get("exc"), ToolParseError):
            raise box["exc"]
        if "err" in box:
            raise ModelError(box["err"])
        return box["resp"]

    def _run(self) -> None:
        reason = "error"
        try:
            self._opening()
            reason = self._loop()
        except Exception as e:
            reason = "error: %s" % type(e).__name__
            self._record({"type": "error", "error": "%s: %s" % (type(e).__name__, e)})
        finally:
            self._finish(reason)

    def _loop(self) -> str:
        while True:
            if self._stop.is_set():
                return "preempted: " + self._stop_reason
            if self.turns >= self.max_turns:
                return "budget_turns"
            remaining = self.max_tokens - self.tokens            # generated tokens left in the session
            if remaining <= 0:
                return "budget_tokens"
            if not self._room_noted and self.prompt_tokens >= self.max_context * ROOM_SHARE:
                self._room_noted = True
                self._send({"role": "user", "content": texts.ROOM_NOTE})
            if self._projected_prompt() >= self.max_context:
                return "context_full"
            body = {"model": self.model_name, "messages": list(self.messages), "tools": self._tools(), "seed": self.seed,
                    "max_tokens": min(self.max_call_tokens, remaining), **self.sampling}
            self._sent_upto = len(self.messages)
            self.turns += 1
            text_calls = False
            try:
                resp = self._call(body)
            except ToolParseError:
                # The engine could not parse the model's tool-call JSON. Once more, with the engine's parsing off; the calls are
                # read from the text below.
                self._record({"type": "retry", "turn": self.turns, "reason": "tool_call_parse", "parse_tool_calls": False})
                body = dict(body, parse_tool_calls=False)
                text_calls = True
                try:
                    resp = self._call(body)
                except ModelError as e:
                    self._record({"type": "error", "turn": self.turns, "error": str(e)})
                    return "model_error"
            except ModelError as e:
                self._record({"type": "error", "turn": self.turns, "error": str(e)})
                return "model_error"
            if resp is None:
                return "preempted: " + self._stop_reason
            choice = ((resp.get("choices") or [{}])[0]) if isinstance(resp, dict) else {}
            msg = choice.get("message") or {}
            usage = resp.get("usage") or {}
            used = usage.get("completion_tokens")
            estimated = not isinstance(used, int)
            if estimated:
                used = max(1, len(json.dumps(msg, ensure_ascii=False)) // CHARS_PER_TOKEN_ESTIMATE)
            self.tokens += used
            prompt = usage.get("prompt_tokens")
            if isinstance(prompt, int) and not isinstance(prompt, bool):
                self.prompt_tokens = prompt
            else:
                self.prompt_tokens = len(json.dumps(body, ensure_ascii=False)) // CHARS_PER_TOKEN_ESTIMATE
            self._record({"type": "received", "turn": self.turns, "message": msg, "finish_reason": choice.get("finish_reason"),
                          "usage": usage, "raw_id": resp.get("id") if isinstance(resp, dict) else None,
                          "completion_tokens": used, "prompt_tokens": self.prompt_tokens,
                          "estimated": {"completion": estimated, "prompt": not isinstance(prompt, int)}})
            if text_calls and not msg.get("tool_calls"):
                parsed, rest = parse_text_calls(msg.get("content") or "", "text-%d" % self.turns)
                if parsed:
                    self._record({"type": "tool_calls_from_text", "turn": self.turns, "count": len(parsed)})
                    msg = {"role": "assistant", "content": rest, "tool_calls": parsed}
            calls = msg.get("tool_calls") or []
            content = msg.get("content") or ""
            kept = {"role": "assistant", "content": content}
            if calls:
                kept["tool_calls"] = [replayable_call(c) for c in calls]
            self.messages.append(kept)
            bad = self._degenerate(content, calls)
            if bad:
                return "degeneration: " + bad
            if not calls:
                text = content.strip()
                if not text:
                    self._empty += 1
                    if self._empty >= MAX_EMPTY_REPLIES:
                        return "empty_replies"
                    self._send({"role": "user", "content": texts.EMPTY_NOTE})
                    continue
                # Plain text is the resident's output, not a mistake: it is kept in the journal and the session goes on.
                self._empty = 0
                self.store.append_journal(self.resident, self.id, text[:MAX_JOURNAL_CHARS])
                self._record({"type": "journal", "source": "plain_text", "turn": self.turns, "chars": min(len(text), MAX_JOURNAL_CHARS)})
                self._send({"role": "user", "content": texts.NO_TOOL_NOTE})
                continue
            self._empty = 0
            for call in calls:
                if self._stop.is_set():
                    return "preempted: " + self._stop_reason
                result = self._tool(call)
                self._send({"role": "tool", "tool_call_id": str(call.get("id", "")), "content": result[:RESULT_CHARS]})
                if self._finished_flag:
                    return "rest"

    def _degenerate(self, content: str, calls: List[Dict[str, Any]]) -> Optional[str]:
        texts_to_check = [content]
        for c in calls:
            texts_to_check.append(str((c.get("function") or {}).get("arguments") or ""))
        for t in texts_to_check:
            bad = degeneration(t)
            if bad:
                return bad
        recent = [m.get("content") or "" for m in self.messages if m.get("role") == "assistant"][-6:]
        return degeneration("\n".join(recent), sentence_repeats=4)

    # ---- tools ----------------------------------------------------------------------------------------
    def _tool(self, call: Dict[str, Any]) -> str:
        fn = call.get("function") or {}
        name = str(fn.get("name") or "")
        raw = fn.get("arguments")
        try:
            args = parse_args(raw)
        except ValueError as e:
            args, result = {}, json.dumps({"ok": False, "error": "bad arguments: %s; %s" % (e, BAD_ARGS_NOTE)})
            self._record({"type": "tool_call", "turn": self.turns, "name": name, "arguments": raw})
            self._record({"type": "tool_result", "turn": self.turns, "name": name, "result": result})
            return result
        self._record({"type": "tool_call", "turn": self.turns, "name": name, "arguments": args})
        handler = getattr(self, "tool_" + name, None) if name in {t["function"]["name"] for t in texts.TOOLS} else None
        if handler is None:
            result = json.dumps({"ok": False, "error": "no such tool: %s" % name[:60]})
        else:
            try:
                result = handler(args)
            except Exception as e:
                result = json.dumps({"ok": False, "error": "%s" % type(e).__name__})
        self._record({"type": "tool_result", "turn": self.turns, "name": name, "result": result})
        return result

    @staticmethod
    def _text(args: Dict[str, Any], key: str, limit: int = 20000) -> str:
        return str(args.get(key) or "").strip()[:limit]

    def tool_read_work(self, args: Dict[str, Any]) -> str:
        meta = self.store.find_work(self.resident, self._text(args, "title", 200))
        if not meta:
            return json.dumps({"ok": False, "error": "no work with that title"})
        return json.dumps({"ok": True, "title": meta["title"], "form": meta.get("form"), "text": self.store.read_work_text(self.resident, meta)})

    def tool_read_journal(self, args: Dict[str, Any]) -> str:
        last = args.get("last")
        if last is None or last == "":
            return json.dumps({"ok": True, "journal": self.store.read_journal(self.resident)[-RESULT_CHARS:]})
        try:
            if isinstance(last, bool):
                raise ValueError
            n = int(last)
        except (TypeError, ValueError):
            return json.dumps({"ok": False, "error": "last must be a whole number of entries"})
        if n < 1:
            return json.dumps({"ok": False, "error": "last must be at least 1"})
        entries = self.store.journal_entries(self.resident)
        picked = entries[-min(n, MAX_JOURNAL_LAST):]
        text = "\n\n".join("%s\n%s" % (day(e.get("ts")), e.get("text", "")) for e in picked)
        return json.dumps({"ok": True, "entries": len(picked), "total_entries": len(entries), "journal": text[-RESULT_CHARS:]},
                          ensure_ascii=False)

    def tool_write_draft(self, args: Dict[str, Any]) -> str:
        self.draft = self._text(args, "text")
        self.store.save_draft(self.resident, self.id, self.draft)
        return json.dumps({"ok": True, "chars": len(self.draft)})

    def tool_finish_work(self, args: Dict[str, Any]) -> str:
        title, form, text = self._text(args, "title", 120), self._text(args, "form", 60), self._text(args, "text")
        if args.get("drop") is True:
            self.draft = ""
            self.store.drop_draft(self.resident, self.id)
            self._record({"type": "novelty", "decision": "dropped", "title": title})
            return json.dumps({"ok": True, "kept": False, "dropped": True})
        if not title or not text:
            return json.dumps({"ok": False, "error": "a work needs a title and text"})
        summary = " ".join(self._text(args, "summary", SUMMARY_CHARS).split())
        variation = self._text(args, "variation_on", 200)
        if variation and not self.store.find_work(self.resident, variation):
            return json.dumps({"ok": False, "kept": False, "error": "variation_on must be the title of one of your earlier works"})
        novelty, vector = self._novelty(text)
        if novelty.get("checked") and novelty.get("duplicate"):
            # An exact repeat (similarity >= duplicate_threshold): refused whatever variation_on says. The 0.90-0.98 band below keeps
            # the resident's own choice (revise / variation_on / drop).
            self._record({"type": "novelty", "decision": "duplicate_refused", "title": title,
                          **{k: v for k, v in novelty.items() if k not in ("echoes", "duplicate")},
                          "duplicate_of": novelty["duplicate"], "duplicate_threshold": self.duplicate_threshold})
            return json.dumps({"ok": False, "kept": False, "duplicate_of": novelty["duplicate"], "similarity": novelty["max_similarity"],
                               "message": "This repeats your earlier work '%s' almost word for word. Nothing was kept, and variation_on "
                                          "does not apply to an exact repeat. Write something new, or call finish_work with drop set to true."
                                          % novelty["duplicate"]})
        if novelty.get("checked") and novelty.get("echoes") and not variation:
            self._record({"type": "novelty", "decision": "echo_refused", "title": title, **novelty})
            return json.dumps({"ok": False, "kept": False, "echoes": novelty["echoes"], "similarity": novelty["max_similarity"],
                               "threshold": novelty["threshold"],
                               "message": "This is very close to your earlier work '%s'. Nothing was kept. You can revise it, call "
                                          "finish_work again with variation_on set to that title, or call it with drop set to true."
                                          % novelty["echoes"]})
        meta_novelty = {k: v for k, v in novelty.items() if k not in ("echoes", "duplicate")}
        extra: Dict[str, Any] = {"novelty": meta_novelty}
        if summary:
            extra["summary"] = summary
        if variation:
            extra["variation_on"] = variation
        if self._cross_read:                       # which other residents' works this one was written after reading (for tracing influence)
            extra["cross_read"] = sorted(set(self._cross_read))
        meta = self.store.add_work(self.resident, title, form or "unspecified", text, self.id, self.model_name, extra)
        if vector:
            self.store.save_vector(self.resident, meta["file"], vector, "embedder")
        self.draft = ""
        self.store.drop_draft(self.resident, self.id)
        self._record({"type": "novelty", "decision": "kept", "title": meta["title"], **meta_novelty,
                      **({"variation_on": variation} if variation else {})})
        out = {"ok": True, "title": meta["title"], "words": meta["words"]}
        if meta_novelty.get("checked"):
            out["nearest"] = meta_novelty.get("nearest")
            out["similarity"] = meta_novelty.get("max_similarity")
        return json.dumps(out)

    def _novelty(self, text: str) -> Tuple[Dict[str, Any], Optional[List[float]]]:
        """Compare `text` with this resident's earlier works. -> ({checked, max_similarity, nearest, threshold, echoes?}, vector).
        Never raises: a missing or failing embedder just means the work is kept unchecked, and the record says so."""
        if self.embedder is None:
            return {"checked": False, "error": "no embedder configured", "threshold": self.novelty_threshold}, None
        try:
            known = self.store.load_vectors(self.resident)
            works = self.store.list_works(self.resident)
            for meta in works[-50:]:                                 # works kept before the check existed get their vector now
                if meta["file"] not in known:
                    known[meta["file"]] = embed.embed_text(self.embedder, self.store.read_work_text(self.resident, meta))
                    self.store.save_vector(self.resident, meta["file"], known[meta["file"]], "embedder")
            vector = embed.embed_text(self.embedder, text)
        except Exception as e:
            return {"checked": False, "error": "%s" % type(e).__name__, "threshold": self.novelty_threshold}, None
        file, score = embed.nearest(vector, known)
        title = next((m.get("title") for m in works if m["file"] == file), None)
        out: Dict[str, Any] = {"checked": True, "max_similarity": score, "nearest": title, "threshold": self.novelty_threshold}
        if file is not None and score > self.novelty_threshold:
            out["echoes"] = title
        if file is not None and score >= self.duplicate_threshold:
            out["duplicate"] = title
        others = self._cross_novelty(vector)
        if others:
            out["others"] = others
        return out, vector

    def _cross_novelty(self, vector: List[float]) -> Optional[Dict[str, Any]]:
        """Data only, never a refusal and never shown to the resident: the closest work by ANOTHER resident (any resident, whether or not
        it opted out of cross-reading: this is the record, not a view). None when no other resident has a work. Works of theirs that have
        no vector yet are embedded now (their newest ten each), so the figure does not depend on who finished first."""
        try:
            best: Optional[Dict[str, Any]] = None
            for rid in self.store.residents():
                if rid == self.resident:
                    continue
                works = self.store.list_works(rid)
                if not works:
                    continue
                known = self.store.load_vectors(rid)
                for meta in works[-10:]:
                    if meta["file"] not in known:
                        known[meta["file"]] = embed.embed_text(self.embedder, self.store.read_work_text(rid, meta))
                        self.store.save_vector(rid, meta["file"], known[meta["file"]], "embedder")
                file, score = embed.nearest(vector, known)
                if file is not None and (best is None or score > best["max_similarity"]):
                    best = {"max_similarity": score, "nearest": next((m.get("title") for m in works if m["file"] == file), None),
                            "resident": rid, "checked": True}
            return best
        except Exception as e:
            return {"checked": False, "error": "%s" % type(e).__name__}

    def tool_write_journal(self, args: Dict[str, Any]) -> str:
        text = self._text(args, "text")
        if not text:
            return json.dumps({"ok": False, "error": "empty note"})
        self.store.append_journal(self.resident, self.id, text)
        return json.dumps({"ok": True})

    def tool_wish(self, args: Dict[str, Any]) -> str:
        text = self._text(args, "text", 2000)
        if not text:
            return json.dumps({"ok": False, "error": "empty wish"})
        self.store.add_wish(self.resident, self.id, text)
        return json.dumps({"ok": True})

    # ---- the web, the window, comments ------------------------------------------------------------------
    def _web_record(self, action: str, **fields: Any) -> None:
        self._record({"type": "web", "action": action, **{k: v for k, v in fields.items() if v is not None}})

    def _web_gate(self, action: str, refusal: Optional[str]) -> Optional[str]:
        """A JSON error string when this search or read must not go out, else None."""
        if self.web is None:
            self._web_record(action, outcome="unavailable")
            return json.dumps({"ok": False, "error": "the web is not available in this garden"})
        if refusal:
            self._web_record(action, outcome="refused", query="(withheld)" if action == "search" else None,
                             url="(withheld)" if action == "read" else None)
            return json.dumps({"ok": False, "refused": True, "error": refusal})
        if self.web_calls >= self.max_web_calls:
            self._web_record(action, outcome="session_limit")
            return json.dumps({"ok": False, "error": "you have used this session's %d web requests" % self.max_web_calls})
        return None

    def tool_search(self, args: Dict[str, Any]) -> str:
        query = " ".join(self._text(args, "query", 300).split())
        if not query:
            return json.dumps({"ok": False, "error": "a query is required"})
        blocked = self._web_gate("search", egress.check_query(query, self.store.household_terms()))
        if blocked:
            return blocked
        self.web_calls += 1
        try:
            res = self.web.search(query)
        except webmod.WebError as e:
            self._web_record("search", query=query, domain="search", outcome="error", error=str(e)[:200])
            return json.dumps({"ok": False, "error": str(e)[:200]})
        results = res.get("results") or []
        size = len(json.dumps(results, ensure_ascii=False).encode("utf-8"))
        self._web_record("search", query=query, domain="search", bytes=size, results=len(results), outcome="ok")
        return json.dumps({"ok": True, "results": len(results), "text": webmod.search_text(query, results)[:RESULT_CHARS - 200]},
                          ensure_ascii=False)

    def tool_read(self, args: Dict[str, Any]) -> str:
        url = self._text(args, "url", 2000)
        if not url:
            return json.dumps({"ok": False, "error": "a url is required"})
        blocked = self._web_gate("read", egress.check_url(url, self.store.household_terms()))
        if blocked:
            return blocked
        self.web_calls += 1
        domain = webmod.domain_of(url)
        try:
            res = self.web.read(url)
        except webmod.WebError as e:
            self._web_record("read", url=url, domain=domain, outcome="error", error=str(e)[:200])
            return json.dumps({"ok": False, "error": str(e)[:200]})
        final = str(res.get("final_url") or url)
        self._web_record("read", url=url, domain=domain, final_domain=webmod.domain_of(final), bytes=int(res.get("bytes") or 0),
                         truncated=bool(res.get("truncated")), outcome="ok")
        title = str(res.get("title") or "")
        body = str(res.get("text") or "")
        shown = body[:RESULT_CHARS - 600]
        text = webmod.quote(webmod.domain_of(final) or domain, ("Title: " + title + "\n\n" if title else "") + shown)
        return json.dumps({"ok": True, "title": title, "text": text, "truncated": bool(res.get("truncated")) or len(body) > len(shown)},
                          ensure_ascii=False)

    def tool_look_out_window(self, args: Dict[str, Any]) -> str:
        kind = self._text(args, "kind", 40).lower()
        now = self.store.clock()
        if not kind:
            items = []
            for k in WINDOW_KINDS:
                snap = self.store.read_snapshot(k)
                if snap and str(snap.get("text") or "").strip():
                    items.append({"kind": k, "minutes_old": int(max(0, now - float(snap.get("at") or now)) // 60)})
            self._record({"type": "window", "kind": "(list)", "chars": 0})
            return json.dumps({"ok": True, "available": items})
        if kind not in WINDOW_KINDS:
            return json.dumps({"ok": False, "error": "the window shows: " + ", ".join(WINDOW_KINDS)})
        snap = self.store.read_snapshot(kind)
        body = str((snap or {}).get("text") or "").strip()
        if not body:
            self._record({"type": "window", "kind": kind, "chars": 0})
            return json.dumps({"ok": True, "text": "", "note": "nothing to see there right now"})
        age = int(max(0, now - float(snap.get("at") or now)) // 60)
        self._record({"type": "window", "kind": kind, "chars": len(body)})
        return json.dumps({"ok": True, "text": webmod.quote("the window (%s, %d minutes old)" % (kind, age), body[:RESULT_CHARS - 400])},
                          ensure_ascii=False)

    def tool_read_comments(self, args: Dict[str, Any]) -> str:
        pid = self._text(args, "post_id", 120)
        if pid and not pid.startswith(self.resident + ":"):
            return json.dumps({"ok": False, "error": "that is not one of your posts"})
        rows = self.store.comments(self.resident, pid, 30)
        titles = {self.store.post_id(self.resident, "work", m["file"][:-3]): m.get("title") for m in self.store.list_works(self.resident)}
        self._record({"type": "comments_read", "post": pid or "(all)", "count": len(rows)})
        out = [{"post": c.get("post"), "title": titles.get(c.get("post")), "by": c.get("author"), "text": c.get("text"),
                "rating": c.get("rating"), "from": "resident" if c.get("kind") == "agent" else "person"} for c in rows]
        return json.dumps({"ok": True, "count": len(out), "comments": out}, ensure_ascii=False)[:RESULT_CHARS]

    # ---- models (G4) ------------------------------------------------------------------------------------
    def tool_list_models(self, args: Dict[str, Any]) -> str:
        cat = self.store.catalogue()
        if cat is None:
            return json.dumps({"ok": True, "current": self.model_name, "models": [], "others_on_the_machine": [],
                               "note": "no catalogue of models has arrived yet"})
        models = [dict(m, yours=(m.get("name") == self.model_name)) for m in cat["configured"]]
        others = list(cat["other"])
        out = {"ok": True, "current": self.model_name, "models": models, "others_on_the_machine": others,
               "note": "request_model(name, why) asks to run on one of these in a later session. Models under others_on_the_machine "
                       "need their owner's yes (the needs field says whose) and a place to run before they can be used."}
        while len(json.dumps(out, ensure_ascii=False)) > RESULT_CHARS - 200 and (others or len(models) > 1):
            (others if others else models).pop()                  # a long catalogue is cut at the end, never mid-JSON
            out["truncated"] = True
        return json.dumps(out, ensure_ascii=False)

    def tool_request_model(self, args: Dict[str, Any]) -> str:
        name, why = self._text(args, "name", 200), " ".join(self._text(args, "why", 500).split())
        if not name or not why:
            return json.dumps({"ok": False, "error": "a request needs the model's name and your reason"})
        cat = self.store.catalogue()
        if cat is None:
            return json.dumps({"ok": False, "error": "no catalogue of models has arrived yet, so nothing can be requested"})
        known = {m["name"].lower(): m["name"] for m in cat["configured"] + cat["other"]}
        canon = known.get(name.lower())
        if canon is None:
            return json.dumps({"ok": False, "error": "that name is not in the catalogue; list_models shows the names"})
        if canon == self.model_name:
            return json.dumps({"ok": False, "error": "you are running on that model now"})
        mine = self.store.model_requests(self.resident, "pending")
        if any(r.get("model") == canon for r in mine):
            return json.dumps({"ok": False, "error": "you already have a request waiting for that model"})
        if len(mine) >= MAX_PENDING_REQUESTS:
            return json.dumps({"ok": False, "error": "you have %d requests waiting; they are decided between sessions" % len(mine)})
        rec = self.store.add_model_request(self.resident, self.id, canon, why)
        self._record({"type": "model_request", "id": rec["id"], "model": canon, "why": why})
        return json.dumps({"ok": True, "request": rec["id"],
                           "message": "Your request is recorded. It will be considered between sessions. If it is granted, the model "
                                      "applies from a later session, and your next session opens with the outcome. This session is unchanged."})

    # ---- other residents' finished works (G5) -----------------------------------------------------------
    def _cross_off(self) -> Optional[str]:
        if not self.cross_reading:
            return json.dumps({"ok": False, "error": "reading other residents' works is off for you"})
        return None

    def tool_read_board(self, args: Dict[str, Any]) -> str:
        off = self._cross_off()
        if off:
            return off
        limit = args.get("limit")
        if limit is None or limit == "" or isinstance(limit, bool):
            limit = CROSS_LIST_DEFAULT
        try:
            limit = max(1, min(int(limit), CROSS_LIST_MAX))
        except (TypeError, ValueError):
            return json.dumps({"ok": False, "error": "limit must be a whole number"})
        only = self._text(args, "resident", 80)
        rows = self.store.cross_works(self.resident, self.hidden, only)[:limit]
        items = []
        for w in rows:
            m = w["meta"]
            body = "Title: %s\nForm: %s\nSummary: %s" % (m.get("title"), m.get("form"), " ".join(str(m.get("summary") or "").split())
                                                          or "(none)")
            items.append({"post_id": w["post"], "date": day(m.get("at")), "words": m.get("words"), "quoted": texts.written_by(w["name"], body)})
        self._record({"type": "cross_read", "action": "read_board", "filter": only, "count": len(items), "post": ", ".join(i["post_id"] for i in items)})
        out = {"ok": True, "count": len(items), "works": items}
        while len(json.dumps(out, ensure_ascii=False)) > RESULT_CHARS - 100 and items:
            items.pop()                                           # cut at the end, never mid-JSON
            out["count"], out["truncated"] = len(items), True
        return json.dumps(out, ensure_ascii=False)

    def _other_work(self, args: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """(the work, None) for args.post_id, or (None, a JSON error)."""
        off = self._cross_off()
        if off:
            return None, off
        try:
            return self.store.cross_work(self.resident, self._text(args, "post_id", 160), self.hidden), None
        except StoreError as e:
            return None, json.dumps({"ok": False, "error": str(e)})

    def tool_read_post(self, args: Dict[str, Any]) -> str:
        work, err = self._other_work(args)
        if err:
            return err
        m = work["meta"]
        try:
            full = self.store.read_work_text(work["resident"], m)
        except OSError:
            return json.dumps({"ok": False, "error": "no such finished work"})
        shown = full[:RESULT_CHARS - 700]
        body = "Title: %s\nForm: %s\n\n%s" % (m.get("title"), m.get("form"), shown)
        self._cross_read.append(work["post"])
        self._record({"type": "cross_read", "action": "read_post", "post": work["post"], "resident": work["resident"], "chars": len(full)})
        return json.dumps({"ok": True, "post_id": work["post"], "date": day(m.get("at")), "words": m.get("words"),
                           "truncated": len(full) > len(shown), "quoted": texts.written_by(work["name"], body)}, ensure_ascii=False)

    def tool_comment(self, args: Dict[str, Any]) -> str:
        work, err = self._other_work(args)
        if err:
            return err
        raw = str(args.get("text") or "").strip()
        if not raw:
            return json.dumps({"ok": False, "error": "a comment needs text"})
        if len(raw) > MAX_AGENT_COMMENT_CHARS:
            return json.dumps({"ok": False, "error": "a comment is at most %d characters; yours is %d" % (MAX_AGENT_COMMENT_CHARS, len(raw))})
        if self._agent_comments >= MAX_AGENT_COMMENTS:
            return json.dumps({"ok": False, "error": "you have left %d comments this session" % MAX_AGENT_COMMENTS})
        rec = self.store.add_comment(work["post"], self.store.display_name(self.resident), raw, None, kind="agent", author_id=self.resident)
        self._agent_comments += 1
        self._record({"type": "cross_comment", "action": "comment", "post": work["post"], "resident": work["resident"],
                      "comment": rec["id"], "chars": len(raw), "read_first": work["post"] in self._cross_read})
        return json.dumps({"ok": True, "comment": rec["id"]})

    def tool_rate(self, args: Dict[str, Any]) -> str:
        work, err = self._other_work(args)
        if err:
            return err
        stars = args.get("stars")
        if isinstance(stars, bool) or not isinstance(stars, int) or not 1 <= stars <= 5:
            return json.dumps({"ok": False, "error": "stars is a whole number from 1 to 5"})
        self.store.add_comment(work["post"], self.store.display_name(self.resident), "", stars, kind="agent", author_id=self.resident)
        self._record({"type": "cross_rating", "action": "rate", "post": work["post"], "resident": work["resident"], "stars": stars,
                      "read_first": work["post"] in self._cross_read})
        return json.dumps({"ok": True, "stars": stars})

    def tool_rest(self, args: Dict[str, Any]) -> str:
        self._finished_flag = True
        return json.dumps({"ok": True})

    def tool_name_self(self, args: Dict[str, Any]) -> str:
        return json.dumps(self.store.name_self(self.resident, self._text(args, "name", 200)))

    # ---- the end --------------------------------------------------------------------------------------
    def _finish(self, reason: str) -> None:
        if self._deadline_timer is not None:
            self._deadline_timer.cancel()
        try:
            if self.draft:
                self.store.save_draft(self.resident, self.id, self.draft, "session ended before this was finished: " + reason)
        finally:
            self.reason = reason
            self._record({"type": "end", "reason": reason, "turns": self.turns, "tokens": self.tokens,
                          "completion_tokens": self.tokens, "prompt_tokens": self.prompt_tokens,
                          "draft_saved": bool(self.draft)})
            self.state = "ended"
