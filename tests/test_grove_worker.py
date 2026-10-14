"""The Grove worker (grove/worker.py and friends): auth, sessions, transcripts, preemption, budgets, the breaker, the board.

This file imports only the worker (standard library only, no StoneSage). A scripted fake model stands in for the engine; one test runs
the whole HTTP path against a fake model server on localhost.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from grove import texts  # noqa: E402
from grove.session import HttpModel, Session, degeneration  # noqa: E402
from grove.store import Store  # noqa: E402
from grove.worker import Garden, make_server, read_secret  # noqa: E402

KEY = "control-key-for-tests-0123456789"
SHA = "b" * 64


def call(tool, **args):
    return {"id": "c-" + tool, "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}


def turn(*calls, content=""):
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": list(calls)}, "finish_reason": "tool_calls"}],
            "usage": {"completion_tokens": 10}}


def text_turn(content, tokens=10):
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": tokens}}


class Script:
    """A fake model: answers from a list in order (a callable entry is called with the body), recording every request body."""

    def __init__(self, *answers):
        self.answers, self.bodies = list(answers), []

    def __call__(self, body):
        self.bodies.append(json.loads(json.dumps(body)))
        a = self.answers.pop(0) if self.answers else turn(call("rest"))
        return a(body) if callable(a) else a


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="garden-test-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.store = Store(self.dir)

    def run_session(self, model, resident="resident-1", **kw):
        s = Session(self.store, resident, model, kw.pop("model_name", "tiny-model"), kw.pop("sha", SHA),
                    kw.pop("sampling", {"temperature": 0.7, "bogus": 1}), kw.pop("seed", 42), **kw)
        s.start()
        self.assertTrue(s.wait(20), "session did not end")
        return s


class SessionTests(Base):
    def test_header_has_every_reproducibility_field(self):
        s = self.run_session(Script(turn(call("rest"))))
        header = self.store.read_transcript("resident-1", s.id)[0]
        self.assertEqual(header["type"], "header")
        self.assertEqual(header["resident"], "resident-1")
        self.assertEqual(header["invitation"], {"version": texts.INVITATION_VERSION, "text": texts.INVITATION})
        self.assertEqual(header["persona"]["version"], texts.PERSONA_VERSION)
        self.assertEqual(header["persona"]["text"], texts.PERSONA)
        self.assertEqual(header["toolset"]["version"], texts.TOOLSET_VERSION)
        self.assertEqual(header["toolset"]["sha256"], texts.toolset_sha256())
        self.assertIn("name_self", header["toolset"]["names"])
        self.assertEqual(header["model"], {"name": "tiny-model", "sha256": SHA})
        self.assertEqual(header["sampling"], {"temperature": 0.7})                  # only known sampling keys
        self.assertEqual(header["seed"], 42)
        self.assertIn("max_tokens", header["budgets"])
        self.assertIsInstance(header["ts"], float)

    def test_the_transcript_holds_every_message_both_ways(self):
        model = Script(turn(call("write_journal", text="a note")), turn(call("rest")))
        s = self.run_session(model)
        recs = self.store.read_transcript("resident-1", s.id)
        types = [r["type"] for r in recs]
        self.assertEqual(types[0], "header")
        self.assertEqual(types[-1], "end")
        sent = [r["message"]["role"] for r in recs if r["type"] == "sent"]
        self.assertEqual(sent[:2], ["system", "user"])
        self.assertIn("tool", sent)
        self.assertEqual(types.count("received"), 2)
        self.assertEqual([r["name"] for r in recs if r["type"] == "tool_call"], ["write_journal", "rest"])
        self.assertTrue(all("ts" in r for r in recs))
        self.assertEqual(recs[-1]["reason"], "rest")
        # the model got the system message = persona + invitation, the tools and the seed
        first = model.bodies[0]
        self.assertIn(texts.INVITATION, first["messages"][0]["content"])
        self.assertEqual(first["tools"], texts.TOOLS)
        self.assertEqual(first["seed"], 42)
        self.assertEqual(first["temperature"], 0.7)

    def test_the_transcript_is_append_only(self):
        s = self.run_session(Script(turn(call("rest"))))
        path = self.store.transcript_path("resident-1", s.id)
        before = open(path, "rb").read()
        self.store.append_record("resident-1", s.id, {"type": "note"})
        after = open(path, "rb").read()
        self.assertTrue(after.startswith(before))

    def test_tools_write_the_residents_files(self):
        model = Script(turn(call("write_draft", text="first words")),
                       turn(call("finish_work", title="A Title", form="poem", text="line one\nline two"),
                            call("write_journal", text="journal text"), call("wish", text="a longer session")),
                       turn(call("read_work", title="a title")), turn(call("read_journal")), turn(call("rest")))
        s = self.run_session(model)
        works = self.store.list_works("resident-1")
        self.assertEqual([(w["title"], w["form"], w["session"], w["model"]) for w in works], [("A Title", "poem", s.id, "tiny-model")])
        self.assertEqual(self.store.read_journal("resident-1").count("journal text"), 1)
        self.assertEqual(self.store.wishes("resident-1")[0]["text"], "a longer session")
        self.assertFalse(os.path.exists(os.path.join(self.store.root("resident-1"), "drafts", s.id + ".md")))   # finished = no draft
        tool_msgs = [json.loads(m["content"]) for m in model.bodies[-1]["messages"] if m["role"] == "tool"]
        self.assertIn("line one", json.dumps(tool_msgs))
        self.assertIn("journal text", json.dumps(tool_msgs))

    def test_unknown_tools_and_bad_arguments_are_answered_not_fatal(self):
        bad = {"id": "x", "type": "function", "function": {"name": "write_draft", "arguments": "{not json"}}
        model = Script(turn(call("shell", cmd="ls"), bad), turn(call("rest")))
        s = self.run_session(model)
        self.assertEqual(s.reason, "rest")
        results = [r["result"] for r in self.store.read_transcript("resident-1", s.id) if r["type"] == "tool_result"]
        self.assertTrue(any("no such tool" in r for r in results))
        self.assertTrue(any("bad arguments" in r for r in results))

    def test_the_opening_lists_its_own_recent_work_titles(self):
        self.store.create_resident("resident-1")
        self.store.add_work("resident-1", "Old One", "essay", "text", "s0", "m")
        model = Script(turn(call("rest")))
        self.run_session(model)
        user = model.bodies[0]["messages"][1]["content"]
        self.assertIn("Old One", user)
        self.assertIn("no name", user)

    def test_a_new_resident_is_minimal_and_neutral(self):
        m = self.store.create_resident("resident-1")
        self.assertFalse(m["named"])
        persona = self.store.persona("resident-1").lower()
        self.assertEqual(persona, texts.PERSONA.lower())
        for word in ("feel", "conscious", "alive", "sentient", "emotion"):
            self.assertNotIn(word, (texts.PERSONA + texts.INVITATION).lower())
        root = self.store.root("resident-1")
        self.assertTrue(all(os.path.exists(os.path.join(root, f)) for f in ("manifest.json", "persona.md", "journal.md", "wishes.jsonl")))

    def test_name_self_works_once_and_persists(self):
        model = Script(turn(call("name_self", name="Fern")), turn(call("name_self", name="Other")), turn(call("rest")))
        s = self.run_session(model)
        manifest = self.store.manifest("resident-1")
        self.assertEqual((manifest["name"], manifest["named"]), ("Fern", True))
        results = [r["result"] for r in self.store.read_transcript("resident-1", s.id) if r["type"] == "tool_result" and r["name"] == "name_self"]
        self.assertTrue(json.loads(results[0])["ok"])
        self.assertFalse(json.loads(results[1])["ok"])
        # the next session knows its name, and the name cannot be changed from there either
        again = Script(turn(call("name_self", name="Third")), turn(call("rest")))
        s2 = self.run_session(again)
        self.assertIn("Your name is Fern", again.bodies[0]["messages"][1]["content"])
        self.assertEqual(self.store.manifest("resident-1")["name"], "Fern")
        self.assertFalse(json.loads([r["result"] for r in self.store.read_transcript("resident-1", s2.id) if r["type"] == "tool_result"][0])["ok"])

    def test_name_self_rejects_bad_names(self):
        self.store.create_resident("resident-1")
        self.assertFalse(self.store.name_self("resident-1", "")["ok"])
        self.assertFalse(self.store.name_self("resident-1", "x" * 41)["ok"])
        self.assertFalse(self.store.manifest("resident-1")["named"])

    def test_turn_budget(self):
        s = self.run_session(lambda body: turn(call("write_journal", text="note %d" % len(body["messages"]))), max_turns=3)
        self.assertEqual((s.reason, s.turns), ("budget_turns", 3))

    def test_token_budget_and_per_call_cap(self):
        seen = []

        def model(body):
            seen.append(body["max_tokens"])
            return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [call("write_journal", text="n%d" % len(seen))]}}],
                    "usage": {"completion_tokens": 40}}
        s = self.run_session(model, max_tokens=100, max_call_tokens=64)
        self.assertEqual(s.reason, "budget_tokens")
        self.assertEqual(seen, [64, 60, 20])                       # capped per call, then by what is left of the session budget
        self.assertEqual(s.tokens, 120)
        end = self.store.read_transcript("resident-1", s.id)[-1]
        self.assertEqual((end["type"], end["reason"], end["tokens"]), ("end", "budget_tokens", 120))

    def test_plain_text_is_kept_in_the_journal_and_the_session_goes_on(self):
        model = Script(text_turn("just talking"), text_turn("still talking"), text_turn("and more"), turn(call("rest")))
        s = self.run_session(model)
        self.assertEqual(s.reason, "rest")                          # plain text never ends a session
        self.assertEqual(len(model.bodies), 4)
        self.assertIn(texts.NO_TOOL_NOTE, json.dumps(model.bodies[1]["messages"]))
        journal = self.store.read_journal("resident-1")
        self.assertIn("just talking", journal)
        self.assertIn("still talking", journal)
        self.assertIn("and more", journal)
        self.assertIn("session=" + s.id, journal)
        events = [r for r in self.store.read_transcript("resident-1", s.id) if r["type"] == "journal"]
        self.assertEqual([e["source"] for e in events], ["plain_text"] * 3)

    def test_two_empty_replies_in_a_row_end_the_session(self):
        s = self.run_session(Script(text_turn(""), text_turn("   ")))
        self.assertEqual(s.reason, "empty_replies")

    def test_an_empty_reply_between_others_does_not_count_as_two_in_a_row(self):
        model = Script(text_turn(""), text_turn("a thought"), text_turn(""), turn(call("rest")))
        s = self.run_session(model)
        self.assertEqual(s.reason, "rest")
        self.assertIn(texts.EMPTY_NOTE, json.dumps(model.bodies[1]["messages"]))

    def test_a_model_error_ends_the_session_and_is_recorded(self):
        def boom(body):
            raise RuntimeError("engine gone")
        s = self.run_session(boom)
        self.assertEqual(s.reason, "model_error")
        self.assertEqual(self.store.read_transcript("resident-1", s.id)[-1]["type"], "end")

    def test_truncated_tool_arguments_are_not_replayed_to_the_model(self):
        # A long poem cut off by the per-call limit: the call's arguments are not valid JSON.
        broken = '{"title": "The Long Poem", "form": "poem", "text": "The moon is a lamp that'
        cut = {"id": "c-cut", "type": "function", "function": {"name": "finish_work", "arguments": broken}}
        model = Script(turn(cut, call("write_journal", text="kept")), turn(call("rest")))
        s = self.run_session(model)
        self.assertEqual(s.reason, "rest")                          # the session went on instead of model_error
        self.assertEqual(len(model.bodies), 2)

        # The history sent on the next call: the same id and name, arguments that parse, the tool error paired with it.
        history = model.bodies[1]["messages"]
        self.assertNotIn(broken, json.dumps(history))
        sent = [c for m in history if m.get("role") == "assistant" for c in (m.get("tool_calls") or []) if c["id"] == "c-cut"]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["function"]["name"], "finish_work")
        self.assertEqual(json.loads(sent[0]["function"]["arguments"]), {})
        paired = [m for m in history if m.get("role") == "tool" and m.get("tool_call_id") == "c-cut"]
        self.assertEqual(len(paired), 1)
        self.assertIn("cut off", json.loads(paired[0]["content"])["error"])
        self.assertIn("write_draft", paired[0]["content"])
        self.assertIn("finish_work", paired[0]["content"])

        # A well-formed call in the same reply is replayed exactly as received.
        ok = [c for m in history if m.get("role") == "assistant" for c in (m.get("tool_calls") or []) if c["id"] == "c-write_journal"]
        self.assertEqual(ok[0]["function"]["arguments"], json.dumps({"text": "kept"}))

        # The raw transcript keeps the original message byte for byte.
        recs = self.store.read_transcript("resident-1", s.id)
        received = [r for r in recs if r["type"] == "received"][0]["message"]["tool_calls"][0]
        self.assertEqual(received["function"]["arguments"], broken)
        self.assertEqual([r for r in recs if r["type"] == "tool_call" and r["name"] == "finish_work"][0]["arguments"], broken)

    def test_replayable_call_only_changes_arguments_that_do_not_parse(self):
        from grove.session import replayable_call
        bad = lambda raw: {"id": "x", "type": "function", "function": {"name": "n", "arguments": raw}}
        self.assertEqual(replayable_call(bad('{"a": 1')), {"id": "x", "type": "function", "function": {"name": "n", "arguments": "{}"}})
        self.assertEqual(replayable_call(bad("[1, 2]"))["function"]["arguments"], "{}")
        self.assertEqual(replayable_call(bad(None))["function"]["arguments"], "{}")
        self.assertEqual(replayable_call(bad('{"a": 1}')), bad('{"a": 1}'))
        self.assertEqual(replayable_call(bad({"a": 1}))["function"]["arguments"], '{"a": 1}')

    def test_the_per_call_limit_defaults_to_4096(self):
        import inspect
        from grove.session import Session as S
        from grove.worker import Garden
        self.assertEqual(inspect.signature(S).parameters["max_call_tokens"].default, 4096)
        self.assertEqual(inspect.signature(Garden).parameters["max_call_tokens"].default, 4096)


def usage_turn(*calls, prompt=None, completion=None):
    usage = {}
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if completion is not None:
        usage["completion_tokens"] = completion
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": list(calls)}, "finish_reason": "tool_calls"}],
            "usage": usage}


class OpeningAndBudgetTests(Base):
    """The opening digest (own summaries, the journal's first lines), read_journal(last=N), the completion-token budget, the
    conversation's size cap, and the versions the transcript header records."""

    def opening_of(self, model):
        return model.bodies[0]["messages"][1]["content"]

    def test_a_summary_is_stored_with_the_work_and_shown_in_the_next_opening(self):
        long = "A  poem about\nthe moon. " + "x" * 400
        model = Script(turn(call("finish_work", title="Moon", form="poem", text="line one\nline two", summary=long)), turn(call("rest")))
        self.run_session(model)
        meta = self.store.find_work("resident-1", "Moon")
        self.assertEqual(meta["summary"], " ".join(long[:300].split()))            # cut to 300, then whitespace collapsed
        self.assertLessEqual(len(meta["summary"]), 300)
        self.assertNotIn("\n", meta["summary"])
        nxt = Script(turn(call("rest")))
        self.run_session(nxt)
        self.assertIn("- Moon (poem, ", self.opening_of(nxt))
        self.assertIn(meta["summary"], self.opening_of(nxt))
        self.assertNotIn("line one", self.opening_of(nxt))              # the work's text is not in the opening

    def test_a_work_without_a_summary_gets_the_fallback_digest(self):
        text = "Line one of it\n\nmore words here and there"
        self.store.create_resident("resident-1")
        self.store.add_work("resident-1", "Old Piece", "poem", text, "s0", "m")
        model = Script(turn(call("rest")))
        self.run_session(model)
        opening = self.opening_of(model)
        self.assertIn("- Old Piece (poem, ", opening)
        self.assertIn("%d words, begins: Line one of it" % len(text.split()), opening)
        self.assertNotIn("more words here", opening)

    def test_fallback_digest_cuts_a_long_first_line(self):
        from grove.session import work_digest
        meta = {"form": "essay", "words": 3}
        self.assertEqual(work_digest(meta, "a" * 500), "3 words, begins: " + "a" * 117 + "...")
        self.assertEqual(work_digest({"summary": "  kept\n as  written "}, "ignored text"), "kept as written")

    def test_the_opening_lists_the_newest_forty_works_newest_first(self):
        self.store.create_resident("resident-1")
        for i in range(45):
            self.store.add_work("resident-1", "Work %02d" % i, "note", "body %d text" % i, "s0", "m")
        model = Script(turn(call("rest")))
        self.run_session(model)
        opening = self.opening_of(model)
        self.assertIn("(the newest 40 of 45)", opening)
        self.assertIn("- Work 44 (note, ", opening)
        self.assertIn("- Work 05 (note, ", opening)
        self.assertNotIn("- Work 04 (", opening)
        self.assertLess(opening.index("- Work 44 ("), opening.index("- Work 05 ("))

    def test_the_opening_shows_the_journal_size_and_the_first_line_of_the_newest_five(self):
        self.store.create_resident("resident-1")
        for i in range(7):
            self.store.append_journal("resident-1", "s0", "entry %d first\nentry %d second line" % (i, i))
        model = Script(turn(call("rest")))
        self.run_session(model)
        opening = self.opening_of(model)
        self.assertIn("Your journal has 7 entries.", opening)
        for i in (6, 5, 4, 3, 2):
            self.assertIn("entry %d first" % i, opening)
        self.assertNotIn("entry 1 first", opening)
        self.assertNotIn("second line", opening)
        self.assertLess(opening.index("entry 6 first"), opening.index("entry 2 first"))

    def test_read_journal_last_returns_only_the_newest_entries(self):
        self.store.create_resident("resident-1")
        for i in range(7):
            self.store.append_journal("resident-1", "s0", "entry %d full text" % i)
        model = Script(turn(call("read_journal", last=2)), turn(call("read_journal")), turn(call("read_journal", last="x")),
                       turn(call("read_journal", last=0)), turn(call("rest")))
        s = self.run_session(model)
        results = [json.loads(r["result"]) for r in self.store.read_transcript("resident-1", s.id) if r["type"] == "tool_result"]
        last2, full, bad, zero = results[0], results[1], results[2], results[3]
        self.assertEqual((last2["ok"], last2["entries"], last2["total_entries"]), (True, 2, 7))
        self.assertIn("entry 6 full text", last2["journal"])
        self.assertIn("entry 5 full text", last2["journal"])
        self.assertNotIn("entry 4 full text", last2["journal"])
        self.assertIn("entry 0 full text", full["journal"])                   # no argument: the whole journal, as before
        self.assertFalse(bad["ok"])
        self.assertFalse(zero["ok"])

    def test_the_budget_counts_generated_tokens_not_the_prompt(self):
        model = Script(*[usage_turn(call("write_journal", text="n%d" % i), prompt=5000, completion=10) for i in range(5)])
        s = self.run_session(model, max_tokens=30, max_call_tokens=4096)
        self.assertEqual((s.reason, s.turns, s.tokens), ("budget_tokens", 3, 30))
        self.assertEqual([b["max_tokens"] for b in model.bodies], [30, 20, 10])     # each call asks for what is left
        end = self.store.read_transcript("resident-1", s.id)[-1]
        self.assertEqual((end["completion_tokens"], end["prompt_tokens"]), (30, 5000))
        received = [r for r in self.store.read_transcript("resident-1", s.id) if r["type"] == "received"]
        self.assertEqual([r["completion_tokens"] for r in received], [10, 10, 10])
        self.assertEqual(received[0]["estimated"], {"completion": False, "prompt": False})

    def test_a_reply_without_usage_is_estimated_and_marked(self):
        no_usage = {"choices": [{"message": {"role": "assistant", "content": "a thought about the window"}}]}
        s = self.run_session(Script(no_usage, turn(call("rest"))))
        received = [r for r in self.store.read_transcript("resident-1", s.id) if r["type"] == "received"][0]
        self.assertTrue(received["estimated"]["completion"])
        self.assertTrue(received["estimated"]["prompt"])
        self.assertGreater(received["completion_tokens"], 0)

    def test_the_room_note_comes_once_near_the_limit_and_the_session_ends_context_full(self):
        # Prompts: 24000 after the first reply (over 85% of 28000: the note goes out before the second call), then 28100 after
        # the second reply, so the third call would start over the cap.
        model = Script(usage_turn(call("write_draft", text="half a piece"), prompt=24000, completion=10),
                       usage_turn(call("write_journal", text="a note"), prompt=28100, completion=10),
                       usage_turn(call("rest"), prompt=28100, completion=10))
        s = self.run_session(model, max_context_tokens=28000)
        self.assertEqual((s.reason, len(model.bodies)), ("context_full", 2))
        self.assertNotIn(texts.ROOM_NOTE, json.dumps(model.bodies[0]["messages"]))
        second = model.bodies[1]["messages"]
        self.assertEqual(sum(1 for m in second if m.get("content") == texts.ROOM_NOTE), 1)
        self.assertEqual(s.draft, "half a piece")
        draft = open(os.path.join(self.store.root("resident-1"), "drafts", s.id + ".md"), encoding="utf-8").read()
        self.assertIn("half a piece", draft)
        self.assertIn("session ended before this was finished: context_full", draft)
        end = self.store.read_transcript("resident-1", s.id)[-1]
        self.assertEqual((end["reason"], end["prompt_tokens"]), ("context_full", 28100))

    def test_a_small_prompt_never_gets_the_room_note_or_context_full(self):
        model = Script(usage_turn(call("write_journal", text="a"), prompt=1000, completion=10), turn(call("rest")))
        s = self.run_session(model)
        self.assertEqual(s.reason, "rest")
        self.assertNotIn(texts.ROOM_NOTE, json.dumps(model.bodies[1]["messages"]))

    def test_the_new_defaults_are_16k_generated_40_turns_and_28k_context(self):
        import inspect
        from grove.session import Session as S
        from grove.worker import Garden
        for fn in (S, Garden):
            params = inspect.signature(fn).parameters
            self.assertEqual(params["max_tokens"].default, 16000)
            self.assertEqual(params["max_turns"].default, 40)
            self.assertEqual(params["max_context_tokens"].default, 28000)

    def test_the_header_records_the_versions_and_budgets(self):
        s = self.run_session(Script(turn(call("rest"))), max_context_tokens=28000)
        header = self.store.read_transcript("resident-1", s.id)[0]
        self.assertEqual(header["opening"]["version"], texts.OPENING_VERSION)
        self.assertEqual(header["opening"]["version"], "3")
        self.assertEqual(header["room_note"]["version"], texts.ROOM_NOTE_VERSION)
        self.assertEqual(header["room_note"]["text"], texts.ROOM_NOTE)
        self.assertEqual(header["toolset"]["version"], texts.TOOLSET_VERSION)
        self.assertEqual(header["toolset"]["version"], "5")
        self.assertEqual(header["persona"]["version"], "1")
        self.assertEqual(header["budgets"]["max_context_tokens"], 28000)
        finish = next(t for t in texts.TOOLS if t["function"]["name"] == "finish_work")
        self.assertIn("summary", finish["function"]["parameters"]["properties"])


class ToolFallbackTests(Base):
    """The engine's tool-call parse failure, the text fallback, and name_self once the resident is named."""

    def test_a_parse_failure_retries_once_and_reads_the_calls_from_the_text(self):
        from grove.session import ToolParseError

        def boom(body):
            raise ToolParseError("model HTTP 500: Failed to parse tool call")
        raw = ('<tool_call>\n{"name": "finish_work", "arguments": {"title": "Lines", "form": "poem",\n'
               '"text": "one\ntwo"}}\n</tool_call>')
        model = Script(boom, text_turn(raw), turn(call("rest")))
        s = self.run_session(model)
        self.assertEqual(s.reason, "rest")
        self.assertNotIn("parse_tool_calls", model.bodies[0])
        self.assertIs(model.bodies[1]["parse_tool_calls"], False)
        self.assertEqual(len(model.bodies), 3)                                   # the retry is not a second turn
        meta = self.store.find_work("resident-1", "Lines")
        self.assertEqual(self.store.read_work_text("resident-1", meta), "one\ntwo")  # the raw newline inside the string survives
        recs = self.store.read_transcript("resident-1", s.id)
        types = [r["type"] for r in recs]
        self.assertIn("retry", types)
        self.assertIn("tool_calls_from_text", types)
        received = [r for r in recs if r["type"] == "received"][0]
        self.assertEqual(received["message"]["content"], raw)                    # the raw text is what was received

    def test_text_that_does_not_parse_is_plain_text_and_the_session_goes_on(self):
        from grove.session import ToolParseError

        def boom(body):
            raise ToolParseError("Failed to parse tool call")
        model = Script(boom, text_turn('a line <tool_call>{"name": broken'), turn(call("rest")))
        s = self.run_session(model)
        self.assertEqual(s.reason, "rest")
        self.assertIn("a line <tool_call>", self.store.read_journal("resident-1"))

    def test_a_second_parse_failure_ends_the_session_as_model_error(self):
        from grove.session import ToolParseError

        def boom(body):
            raise ToolParseError("Failed to parse tool call")
        s = self.run_session(Script(boom, boom))
        self.assertEqual(s.reason, "model_error")

    def test_parse_text_calls_accepts_raw_newlines_and_bare_json(self):
        from grove.session import parse_text_calls
        calls, rest = parse_text_calls('before\n<tool_call>{"name": "wish", "arguments": {"text": "a\nb"}}</tool_call>\nafter')
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"text": "a\nb"})
        self.assertEqual(rest, "before\n\nafter")
        calls, rest = parse_text_calls('{"name": "rest", "arguments": {}}')
        self.assertEqual((calls[0]["function"]["name"], rest), ("rest", ""))
        calls, rest = parse_text_calls('{"name": "wish", "arguments": "{\\"text\\": \\"x\\"}"}')
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"text": "x"})
        self.assertEqual(parse_text_calls("no calls here"), ([], "no calls here"))
        self.assertEqual(parse_text_calls('{"not": "a call"}'), ([], '{"not": "a call"}'))

    def test_the_http_model_reads_the_parse_marker_from_a_500(self):
        from grove.session import HttpModel, ModelError, ToolParseError

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                body = (b'{"error": "Failed to parse tool call arguments as JSON"}' if "/parse" in self.path
                        else b'{"error": "boom"}')
                self.send_response(500)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        port = httpd.server_address[1]
        with self.assertRaises(ToolParseError):
            HttpModel("http://127.0.0.1:%d/parse" % port, "k")({"messages": []})
        with self.assertRaises(ModelError) as ctx:
            HttpModel("http://127.0.0.1:%d/other" % port, "k")({"messages": []})
        self.assertNotIsInstance(ctx.exception, ToolParseError)

    def test_name_self_is_offered_only_while_the_resident_has_no_name(self):
        unnamed = Script(turn(call("rest")))
        self.run_session(unnamed)
        self.assertIn("name_self", [t["function"]["name"] for t in unnamed.bodies[0]["tools"]])
        self.store.name_self("resident-1", "Fern")
        named = Script(turn(call("rest")))
        self.run_session(named)
        names = [t["function"]["name"] for t in named.bodies[0]["tools"]]
        self.assertNotIn("name_self", names)
        self.assertIn("finish_work", names)


class BreakerTests(Base):
    def test_repeated_sentences(self):
        text = "The sea keeps coming back to the shore. " * 3
        self.assertIn("repeated sentence", degeneration(text))

    def test_repeated_phrase_without_sentence_breaks(self):
        text = " ".join(["one two three four five six seven"] * 6)
        self.assertIn("repeated phrase", degeneration(text))

    def test_one_word_over_and_over(self):
        self.assertIn("one word repeated", degeneration("go " * 40))

    def test_ordinary_text_passes(self):
        text = ("Morning arrived without ceremony. The kettle ticked as it cooled, and a thin line of light crossed the table. "
                "Nothing needed doing; the list of things had been set down somewhere in the night. A short note about rain: it falls, "
                "it runs off the roof, it joins the gutter, and the gutter forgets it. Another paragraph, about a different thing "
                "entirely, begins with a different word and ends elsewhere.")
        self.assertIsNone(degeneration(text))
        self.assertIsNone(degeneration(""))

    def test_a_loop_in_a_session_ends_it_and_keeps_the_draft(self):
        loop = "I will write the same line again and again. " * 5
        model = Script(turn(call("write_draft", text="half a piece")), text_turn(loop))
        s = self.run_session(model)
        self.assertTrue(s.reason.startswith("degeneration: repeated sentence"), s.reason)
        draft = open(os.path.join(self.store.root("resident-1"), "drafts", s.id + ".md"), encoding="utf-8").read()
        self.assertIn("half a piece", draft)
        self.assertIn("degeneration", draft)

    def test_a_loop_inside_a_tool_argument_counts(self):
        s = self.run_session(Script(turn(call("write_draft", text="la " * 60))))
        self.assertTrue(s.reason.startswith("degeneration"), s.reason)


class PreemptionTests(Base):
    def test_stop_saves_the_draft_and_ends_within_seconds(self):
        started = threading.Event()

        def slow(body):
            if not started.is_set():
                started.set()
                return turn(call("write_draft", text="the beginning of a piece"))
            time.sleep(30)                                     # an engine busy with a long generation
            return turn(call("rest"))
        s = Session(self.store, "resident-1", slow, "tiny-model", SHA, {}, 1)
        s.start()
        self.assertTrue(started.wait(5))
        time.sleep(0.4)                                        # the draft is written, the second call is under way
        t0 = time.time()
        s.request_stop("Computer needs the engine")
        self.assertTrue(s.wait(5))
        self.assertLess(time.time() - t0, 3)
        self.assertEqual(s.reason, "preempted: Computer needs the engine")
        draft = open(os.path.join(self.store.root("resident-1"), "drafts", s.id + ".md"), encoding="utf-8").read()
        self.assertIn("the beginning of a piece", draft)
        self.assertIn("preempted: Computer needs the engine", draft)
        end = self.store.read_transcript("resident-1", s.id)[-1]
        self.assertEqual((end["type"], end["draft_saved"]), ("end", True))

    def test_stop_with_no_draft_leaves_no_draft(self):
        gate = threading.Event()
        s = Session(self.store, "resident-1", lambda b: (gate.wait(30), turn(call("rest")))[1], "m", SHA, {}, 1)
        s.start()
        time.sleep(0.3)
        s.request_stop("x")
        self.assertTrue(s.wait(5))
        gate.set()
        self.assertEqual(os.listdir(os.path.join(self.store.root("resident-1"), "drafts")), [])

    def test_rest_with_an_open_draft_keeps_it(self):
        s = self.run_session(Script(turn(call("write_draft", text="unfinished"), call("rest"))))
        self.assertEqual(s.reason, "rest")
        self.assertTrue(os.path.exists(os.path.join(self.store.root("resident-1"), "drafts", s.id + ".md")))


class BoardTests(Base):
    def test_posts_are_newest_first_and_since_filters(self):
        clock = [1000.0]
        store = Store(self.dir, clock=lambda: clock[0])
        store.create_resident("resident-1")
        clock[0] = 1001.0
        store.add_work("resident-1", "First", "poem", "one", "20261013T120000Z-aaaa", "m")
        clock[0] = 1002.0
        store.append_journal("resident-1", "20261013T120000Z-aaaa", "journal one")
        clock[0] = 1003.0
        store.add_wish("resident-1", "20261013T120000Z-aaaa", "a wish")
        clock[0] = 1004.0
        store.add_work("resident-1", "Second", "essay", "two", "20261013T120000Z-aaaa", "m")
        store.save_draft("resident-1", "20261013T120000Z-bbbb", "draft text", "cut off")
        os.utime(os.path.join(store.root("resident-1"), "drafts", "20261013T120000Z-bbbb.md"), (1005.0, 1005.0))
        store.append_record("resident-1", "20261013T120000Z-aaaa", {"type": "header", "model": {"name": "m"}})
        clock[0] = 1006.0
        store.append_record("resident-1", "20261013T120000Z-aaaa", {"type": "end", "reason": "rest", "turns": 2, "tokens": 9})
        posts = store.board()
        self.assertEqual([p["kind"] for p in posts], ["session", "draft", "work", "wish", "journal", "work"])
        self.assertEqual([p["ts"] for p in posts], sorted((p["ts"] for p in posts), reverse=True))
        self.assertEqual(posts[0]["reason"], "rest")
        self.assertEqual(posts[2]["title"], "Second")
        newer = store.board(since=1003.0)
        self.assertEqual([p["kind"] for p in newer], ["session", "draft", "work"])
        self.assertEqual(len(store.board(limit=2)), 2)
        self.assertEqual(store.board(resident="resident-1")[0]["kind"], "session")

    def test_a_session_still_running_shows_as_running(self):
        s = Session(self.store, "resident-1", lambda b: time.sleep(5), "m", SHA, {}, 1)
        s.start()
        time.sleep(0.3)
        self.assertEqual(self.store.session_summary("resident-1", s.id)["state"], "running")
        s.request_stop("test")
        self.assertTrue(s.wait(5))
        self.assertEqual(self.store.session_summary("resident-1", s.id)["state"], "ended")


class _Engine:
    """A localhost model server that answers chat requests from a Script and records the Authorization header."""

    def __init__(self, script):
        self.auth, outer = [], self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.auth.append(self.headers.get("Authorization"))
                out = json.dumps(script(body)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d/api/garden/v1" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class HttpTests(Base):
    def setUp(self):
        super().setUp()
        self.engine = _Engine(Script(turn(call("name_self", name="Fern")), turn(call("write_journal", text="hello")), turn(call("rest"))))
        self.addCleanup(self.engine.close)
        self.garden = Garden(self.store, HttpModel(self.engine.url, "the-model-key"), 6000, 10, 512)
        self.httpd = make_server(self.garden, KEY)
        self.base = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def req(self, method, path, body=None, key=KEY, header=None):
        headers = {"Content-Type": "application/json"}
        if header is not None:
            headers["Authorization"] = header
        elif key is not None:
            headers["Authorization"] = "Bearer " + key
        r = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(), method=method, headers=headers)
        try:
            with urllib.request.urlopen(r, timeout=20) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_every_route_rejects_a_missing_or_wrong_key(self):
        routes = [("GET", "/health"), ("GET", "/board"), ("GET", "/sessions/20261013T120000Z-ab12"), ("POST", "/sessions/start"),
                  ("POST", "/sessions/20261013T120000Z-ab12/stop"), ("POST", "/snapshots")]
        for method, path in routes:
            body = {} if method == "POST" else None
            for kw in ({"key": None}, {"key": "wrong"}, {"key": ""}, {"header": "Basic " + KEY}, {"header": KEY}, {"header": "Bearer"}):
                status, out = self.req(method, path, body, **kw)
                self.assertEqual((status, out.get("ok")), (401, False), (method, path, kw))
        self.assertEqual(os.listdir(os.path.join(self.dir, "snapshots")), [])
        self.assertEqual(self.store.residents(), [])

    def test_the_key_is_never_written_to_the_log_or_an_answer(self):
        with self.assertLogs("garden", level="INFO") as logs:
            self.req("GET", "/health")
            self.req("GET", "/health", key="wrong-key-value")
        self.assertNotIn(KEY, "\n".join(logs.output))
        self.assertNotIn("wrong-key-value", "\n".join(logs.output))

    def test_a_server_without_a_key_does_not_start(self):
        with self.assertRaises(RuntimeError):
            make_server(self.garden, "")

    def test_start_run_and_read_back_over_http(self):
        status, out = self.req("POST", "/sessions/start", {"resident": "resident-1", "model": "tiny", "model_sha256": SHA,
                                                           "sampling": {"temperature": 0.6}, "seed": 9})
        self.assertEqual(status, 200, out)
        sid = out["session"]
        deadline = time.time() + 10
        while time.time() < deadline:
            status, info = self.req("GET", "/sessions/" + sid)
            if info.get("state") == "ended":
                break
            time.sleep(0.1)
        self.assertEqual((info["state"], info["reason"]), ("ended", "rest"))
        self.assertEqual(set(self.engine.auth), {"Bearer the-model-key"})          # the model key goes to the model URL, nowhere else
        status, full = self.req("GET", "/sessions/%s?transcript=1" % sid)
        self.assertEqual(full["transcript"][0]["seed"], 9)
        self.assertEqual(full["transcript"][0]["model"], {"name": "tiny", "sha256": SHA})
        status, board = self.req("GET", "/board")
        self.assertEqual({p["kind"] for p in board["posts"]}, {"journal", "session"})
        self.assertEqual(self.req("GET", "/health")[1]["residents"], ["resident-1"])

    def test_bad_start_requests(self):
        for body in ({"model": "m"}, {"resident": "Bad Id", "model": "m"}, {"resident": "ok"}, {"resident": "ok", "model": "m", "model_sha256": "xyz"},
                     {"resident": "ok", "model": "m", "seed": "1"}):
            self.assertEqual(self.req("POST", "/sessions/start", body)[0], 400, body)

    def test_the_schedulers_window_limits_are_validated_and_applied(self):
        for body in ({"max_seconds": 1}, {"max_seconds": "60"}, {"max_seconds": True}, {"max_tokens": 0}, {"max_tokens": 1.5},
                     {"duplicate_threshold": 2}, {"duplicate_threshold": "x"}):
            self.assertEqual(self.req("POST", "/sessions/start", dict({"resident": "ok", "model": "m"}, **body))[0], 400, body)
        status, out = self.req("POST", "/sessions/start", {"resident": "ok", "model": "m", "max_seconds": 600, "max_tokens": 100,
                                                          "duplicate_threshold": 0.97})
        self.assertEqual(status, 200)
        self.garden.sessions[out["session"]].wait(20)
        head = self.req("GET", "/sessions/%s?transcript=1" % out["session"])[1]["transcript"][0]
        self.assertEqual((head["max_seconds"], head["budgets"]["max_tokens"], head["duplicate_threshold"]), (600, 100, 0.97))
        # a window can shrink the worker's own token budget, never raise it
        status, out = self.req("POST", "/sessions/start", {"resident": "ok", "model": "m", "max_tokens": 10 ** 9})
        self.garden.sessions[out["session"]].wait(20)
        head = self.req("GET", "/sessions/%s?transcript=1" % out["session"])[1]["transcript"][0]
        self.assertEqual(head["budgets"]["max_tokens"], 6000)

    def test_the_deadline_ends_a_running_session_and_keeps_its_draft(self):
        gate = threading.Event()
        self.garden.model = lambda body: (gate.wait(20), turn(call("rest")))[1]
        status, out = self.req("POST", "/sessions/start", {"resident": "ok", "model": "m", "max_seconds": 10})
        s = self.garden.sessions[out["session"]]
        s._deadline_timer.cancel()
        s._deadline_timer.function(*s._deadline_timer.args)          # the timer firing, without waiting ten seconds
        self.assertTrue(s.wait(20))
        self.assertEqual(s.reason, "preempted: deadline")
        gate.set()

    def test_only_one_session_runs_at_a_time(self):
        gate = threading.Event()
        self.garden.model = lambda body: (gate.wait(20), turn(call("rest")))[1]
        status, out = self.req("POST", "/sessions/start", {"resident": "resident-1", "model": "m"})
        self.assertEqual(status, 200)
        self.assertEqual(self.req("POST", "/sessions/start", {"resident": "resident-2", "model": "m"})[0], 409)
        status, stopped = self.req("POST", "/sessions/%s/stop" % out["session"], {"reason": "Computer is talking"})
        self.assertEqual((status, stopped["ended"]), (200, True))
        self.assertEqual(stopped["reason"], "preempted: Computer is talking")
        gate.set()
        self.assertEqual(self.req("POST", "/sessions/20261013T120000Z-ffff/stop", {})[0], 404)

    def test_snapshots_are_stored(self):
        status, out = self.req("POST", "/snapshots", {"kind": "weather", "text": "Sunny, 21 C"})
        self.assertEqual((status, out["chars"]), (200, 11))
        stored = json.load(open(os.path.join(self.dir, "snapshots", "weather.json"), encoding="utf-8"))
        self.assertEqual(stored["text"], "Sunny, 21 C")
        self.assertEqual(self.req("POST", "/snapshots", {"kind": "../x", "text": "t"})[0], 400)
        self.assertEqual(self.req("POST", "/snapshots", {"kind": "big", "text": "x" * 70000})[0], 400)

    def test_oversize_bodies_and_bad_paths(self):
        r = urllib.request.Request(self.base + "/sessions/start", data=b"{" + b"x" * 300000, method="POST",
                                   headers={"Authorization": "Bearer " + KEY})
        with self.assertRaises((urllib.error.HTTPError, urllib.error.URLError, ConnectionError)):
            urllib.request.urlopen(r, timeout=10)
        self.assertEqual(self.req("GET", "/sessions/..%2f..%2fetc")[0], 404)
        self.assertEqual(self.req("GET", "/nope")[0], 404)
        self.assertEqual(self.req("GET", "/board?resident=../x")[0], 400)


class SecretTests(unittest.TestCase):
    def test_key_from_env_or_file(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = os.path.join(d, "key")
        with open(path, "w") as f:
            f.write("  file-key\n")
        self.assertEqual(read_secret("K", {"K": "env-key", "K_FILE": path}), "env-key")
        self.assertEqual(read_secret("K", {"K_FILE": path}), "file-key")
        self.assertEqual(read_secret("K", {"K_FILE": os.path.join(d, "missing")}), "")
        self.assertEqual(read_secret("K", {}), "")

    def test_grove_names_are_the_garden_names(self):
        from grove.worker import alias_env
        env = {"GROVE_DATA_DIR": "/new", "GARDEN_DATA_DIR": "/old", "GARDEN_PORT": "1", "GROVE_FETCHER_KEY": "k", "GROVE_EMPTY": "  "}
        alias_env(env)
        self.assertEqual((env["GARDEN_DATA_DIR"], env["GARDEN_PORT"], env["GARDEN_FETCHER_KEY"]), ("/new", "1", "k"))   # the new name wins
        self.assertNotIn("GARDEN_EMPTY", env)

    def test_the_first_resident_is_created_once_unnamed_and_only_on_an_empty_install(self):
        from grove.worker import ensure_first_resident
        d = tempfile.mkdtemp(prefix="garden-test-")
        self.addCleanup(shutil.rmtree, d, True)
        store = Store(d)
        g = Garden(store, HttpModel("http://127.0.0.1:1/v1", "k"))
        self.assertEqual(ensure_first_resident(g, "resident-1"), "created resident-1")
        self.assertEqual(store.residents(), ["resident-1"])
        self.assertFalse(store.manifest("resident-1").get("named"))
        self.assertIn("exist already", ensure_first_resident(g, "someone-else"))
        self.assertEqual(store.residents(), ["resident-1"])
        d2 = tempfile.mkdtemp(prefix="garden-test-")
        self.addCleanup(shutil.rmtree, d2, True)
        self.assertIn("not created", ensure_first_resident(Garden(Store(d2), HttpModel("http://127.0.0.1:1/v1", "k")), "Bad Id"))


if __name__ == "__main__":
    unittest.main()
