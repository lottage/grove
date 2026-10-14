"""Garden step G3 (stonesage/garden_scheduler.py, garden_client.py): gate-driven scheduling, per-resident models, preemption.

No LAN, no worker, no engine: a fake idle gate, a fake engine lease, a fake worker and a fake clock. Does not import `garden` (the
worker), so the public split can carry it.
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from stonesage import apps, auth  # noqa: E402
from grove.stonesage_app import client as garden_client, scheduler as gs  # noqa: E402
from grove.stonesage_app.client import ActiveEngine, GardenError  # noqa: E402

T0 = 1_700_000_000.0
GPU = "RX 6600"
OTHER = "RX 6750 XT"
LAYOUTS = {
    "longctx_6600": {"layout": "longctx_6600", "roles": ["vision"], "gpus": [GPU], "resolved": True, "disabled": None},
    "longctx_6750": {"layout": "longctx_6750", "roles": ["coordinator", "worker"], "gpus": [OTHER], "resolved": True, "disabled": None,
                       "stand_in_ready": True},
    "longctx_max": {"layout": "longctx_max", "roles": ["coordinator", "worker", "vision"], "gpus": [OTHER, GPU], "resolved": True,
                    "disabled": None, "stand_in_ready": True},
    "unresolved": {"layout": "unresolved", "roles": [], "gpus": [], "resolved": False, "disabled": None},
}


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s

    def sleep(self, s):
        self.t += s


class FakeLease:
    def __init__(self, clock):
        self.clock, self.lease, self.acquired, self.released = clock, None, [], []
        self.refuse = None
        self.release_ok = True

    def layout_info(self, layout):
        return LAYOUTS.get(layout)

    def acquire(self, layout, ttl_s=3600, reason="", approver="", force=False, yield_to_computer=False):
        if self.refuse:
            return {"ok": False, "error": self.refuse}
        expires = self.clock() + ttl_s
        self.acquired.append({"layout": layout, "ttl_s": ttl_s, "at": self.clock(), "approver": approver, "force": force, "reason": reason,
                              "yield_to_computer": yield_to_computer})
        self.lease = {"id": "L1", "layout": layout, "expires": expires, "state": "active"}
        return {"ok": True, "lease_id": "L1", "expires": expires, "url": "http://engine:8005", "until": "x"}

    def release(self, lease_id=None, why="released", approver=""):
        self.released.append({"at": self.clock(), "why": why, "expires": (self.lease or {}).get("expires"), "id": lease_id})
        self.lease = None
        return {"ok": self.release_ok}


class FakeWorker:
    def __init__(self, active):
        self.active = active
        self.running, self.started, self.stopped, self.n = None, [], [], 0
        self.down = False
        self.refuse = False
        self.residents_list = [{"id": "ada", "paused": False}]
        self.ended = {}

    def health(self):
        if self.down:
            raise GardenError("the garden worker did not answer")
        return {"ok": True, "running": self.running}

    def residents(self):
        if self.down:
            raise GardenError("the garden worker did not answer")
        return {"ok": True, "residents": self.residents_list}

    def start_session(self, body, by="manual"):
        if self.refuse:
            raise GardenError("a session is already running", 409)
        self.n += 1
        sid = "20261013T%06dZ-%04x" % (self.n, self.n)
        self.running = sid
        self.started.append((dict(body), by, sid))
        self.active.set("http://engine/" + body["model"], body["model"], by)
        self.active.bind(sid)
        return {"ok": True, "session": sid, "resident": body["resident"]}

    def stop_session(self, sid, reason=""):
        self.stopped.append((sid, reason))
        self.running = None
        self.active.clear(sid)
        return {"ok": True, "tokens": 321}

    def session(self, sid):
        return self.ended.get(sid, {"state": "ended", "tokens": 500, "reason": "rest"})

    def finish(self, tokens=500, reason="rest"):
        self.ended[self.running] = {"state": "ended", "tokens": tokens, "reason": reason}
        self.running = None


class World:
    """A garden with everything faked. Defaults describe a house where a session may start right now."""

    def __init__(self, test, **sched_cfg):
        self.dir = tempfile.mkdtemp(prefix="garden-sched-")
        test.addCleanup(shutil.rmtree, self.dir, True)
        self.clock = Clock()
        self.gate = {"gpus": {GPU: {"open": True, "reasons": []}, OTHER: {"open": True, "reasons": []}}, "shared_reasons": []}
        self.pending, self.last_turn = 0, None
        self.config = {"garden": {
            "models": {"bonsai": {"layout": "longctx_6600", "engine_url": "http://engine:8005/v1", "model_sha256": "a" * 64},
                       "big": {"layout": "longctx_6750", "engine_url": "http://engine:8005/v1"},
                       "granite-cpu": {"layout": None, "unit": "llama-courage-lite.service", "engine_url": "http://engine:8006/v1"},
                       "broken": {"layout": "unresolved", "engine_url": "http://engine:8005/v1"}},
            "residents": {"ada": {"model": "bonsai"}},
            "schedule": dict({"enabled": True}, **sched_cfg)}}
        self.active = ActiveEngine(self.clock)
        self.lease = FakeLease(self.clock)
        self.worker = FakeWorker(self.active)
        self.trace = []
        self.units = []
        self.health_ok = True
        self.sched = gs.GardenScheduler(
            lambda: self.config, lambda fresh: self.gate, self.lease, self.worker, self.active,
            lambda: self.last_turn, lambda: self.pending, trace=self.trace.append, data_dir=self.dir,
            systemctl=self._systemctl, http_ok=lambda url: self.health_ok, clock=self.clock, sleep=self.clock.sleep)

    def _systemctl(self, verb, unit):
        self.units.append((verb, unit))
        return {"ok": True}

    def events(self, name=None):
        return [r for r in self.trace if r.get("kind") == "garden" and (name is None or r.get("event") == name)]

    def close_gate(self, *reasons, gpu=GPU):
        self.gate["gpus"][gpu] = {"open": False, "reasons": list(reasons)}

    def tick(self, n=1, step=0):
        for _ in range(n):
            self.sched.tick()
            self.clock.advance(step)

    def assign(self, resident, model):
        self.config["garden"]["residents"][resident] = {"model": model}


class Base(unittest.TestCase):
    def world(self, **cfg):
        return World(self, **cfg)


# ---- the gate's history ----------------------------------------------------------------------------------------------------
class HistoryTests(Base):
    def test_transitions_are_recorded_per_gpu_with_why_even_when_the_schedule_is_off(self):
        w = self.world(enabled=False)
        w.tick()
        self.assertEqual(w.sched.state, "disabled")
        w.clock.advance(600)
        w.close_gate("Computer answered 1 min ago", "GPU 40 % busy")
        w.gate["shared_reasons"] = ["Computer answered 1 min ago"]
        w.tick()
        lines = [json.loads(x) for x in open(os.path.join(w.dir, "garden_gate_history.jsonl"), encoding="utf-8")]
        mine = [r for r in lines if r["gpu"] == GPU]
        self.assertEqual([r["event"] for r in mine], ["open", "close"])
        self.assertEqual(mine[1]["opened_at"], T0)
        self.assertAlmostEqual(mine[1]["at"] - mine[0]["at"], 600, places=2)
        self.assertIn("GPU 40 % busy", mine[1]["why"])
        self.assertEqual(w.worker.started, [])          # recording only
        self.assertEqual(w.lease.acquired, [])

    def test_only_changes_are_written_and_a_restart_continues_the_open_opening(self):
        w = self.world(enabled=False)
        w.tick(5, step=20)
        n = len(open(os.path.join(w.dir, "garden_gate_history.jsonl"), encoding="utf-8").readlines())
        self.assertEqual(n, 3)                          # open for the two GPUs and the house, nothing else
        again = gs.GateHistory(os.path.join(w.dir, "garden_gate_history.jsonl"), w.clock)
        self.assertEqual(again.open_since[GPU], T0)
        again.observe({GPU: (True, [])}, w.clock() + 100)
        self.assertEqual(len(open(os.path.join(w.dir, "garden_gate_history.jsonl"), encoding="utf-8").readlines()), 3)
        later = gs.GateHistory(os.path.join(w.dir, "garden_gate_history.jsonl"), w.clock)
        later.observe({GPU: (False, ["busy"])}, w.clock() + 200)
        self.assertTrue(later.closed[GPU][-1]["censored"])       # the first look after a restart: the real end is unknown
        self.assertEqual(later.lengths(GPU), [])

    def test_openings_we_ended_ourselves_and_censored_ones_are_not_samples(self):
        h = gs.GateHistory(None)
        h.observe({GPU: (False, [])}, T0)
        h.observe({GPU: (True, [])}, T0 + 10)
        h.observe({GPU: (False, ["Computer"])}, T0 + 1010)        # natural: 1000 s
        h.observe({GPU: (True, [])}, T0 + 2000)
        h.mark_own_end([GPU], T0 + 3000)
        h.observe({GPU: (False, ["busy"])}, T0 + 3100)           # ours: within 15 min of our own release
        h.observe({GPU: (True, [])}, T0 + 9000)
        h.observe({GPU: (False, ["x"])}, T0 + 9000 + 800)        # natural again, long after the mark
        self.assertEqual(h.lengths(GPU), [1000.0, 800.0])


# ---- the estimate ----------------------------------------------------------------------------------------------------------
class EstimateTests(Base):
    def test_fewer_than_five_comparable_openings_means_the_default(self):
        e = gs.estimate_remaining([3600, 3600, 3600, 3600], 0, 1800)
        self.assertEqual((e["seconds"], e["basis"], e["samples"]), (1800.0, "default", 4))

    def test_median_of_the_remaining_time_of_openings_at_least_as_long(self):
        lengths = [300, 600, 3000, 3600, 4200, 5400, 7200]       # age 1800: the last five qualify -> remaining 1200 1800 2400 3600 5400
        e = gs.estimate_remaining(lengths, 1800, 1800)
        self.assertEqual((e["seconds"], e["basis"], e["samples"]), (2400.0, "history", 5))

    def test_an_opening_exactly_as_long_as_the_age_counts_and_a_shorter_one_does_not(self):
        e = gs.estimate_remaining([1000] * 5 + [999], 1000, 60)
        self.assertEqual((e["samples"], e["seconds"]), (5, 0.0))
        self.assertEqual(gs.estimate_remaining([1000] * 4 + [999], 1000, 60)["basis"], "default")

    def test_a_long_history_decides_the_start_and_a_short_one_forbids_it(self):
        w = self.world()
        # six past openings of ~10 min each: the median remaining at age 0 is 10 min < min_window_min (15)
        for i in range(6):
            w.sched.history.observe({GPU: (True, [])}, T0 - 90000 + i * 5000)
            w.sched.history.observe({GPU: (False, ["Computer"])}, T0 - 90000 + i * 5000 + 600)
        w.tick()
        self.assertEqual(w.lease.acquired, [])
        self.assertIn("expected to last 10 more min", w.sched.reason)
        # six openings of 2 h: the median is 2 h -> start, window capped at max_window_min
        w2 = self.world()
        for i in range(6):
            w2.sched.history.observe({GPU: (True, [])}, T0 - 90000 + i * 9000)
            w2.sched.history.observe({GPU: (False, ["Computer"])}, T0 - 90000 + i * 9000 + 7200)
        w2.tick()
        self.assertEqual(len(w2.lease.acquired), 1)
        self.assertEqual(w2.lease.acquired[0]["ttl_s"], (55 + 10) * 60)
        started = w2.events("scheduled_start")[0]
        self.assertEqual((started["basis"], started["samples"]), ("history", 6))


# ---- when a start is allowed ---------------------------------------------------------------------------------------------
class StartConditionTests(Base):
    def test_the_happy_path_takes_the_lease_and_starts_a_session_with_its_deadline(self):
        w = self.world()
        w.tick()
        self.assertEqual(w.sched.state, "running")
        a = w.lease.acquired[0]
        self.assertEqual((a["layout"], a["approver"], a["force"]), ("longctx_6600", "garden-scheduler", False))
        body, by, sid = w.worker.started[0]
        self.assertEqual((body["resident"], body["model"], by), ("ada", "bonsai", "scheduler"))
        self.assertAlmostEqual(body["max_seconds"], 28 * 60, delta=2)         # 30 min estimate + 10 margin - 12 before the end
        self.assertNotIn("max_tokens", body)                                  # speed not known yet
        self.assertEqual(w.events("scheduled_start")[0]["outcome"], "started")

    def test_each_guard_stops_a_start(self):
        cases = {
            "lease held": (lambda w: setattr(w.lease, "lease", {"id": "Z", "layout": "x", "expires": T0 + 600}), "lease is held"),
            "approval waiting": (lambda w: setattr(w, "pending", 1), "approval"),
            "computer turn 5 min ago": (lambda w: setattr(w, "last_turn", T0 - 300), "Computer answered 5 min ago"),
            "gate closed for the GPU": (lambda w: w.close_gate("vision served a request in the last 5 min"), "idle gate is closed"),
            "gate missing": (lambda w: w.gate["gpus"].pop(GPU), "idle gate is closed"),
            "manual session running": (lambda w: setattr(w.worker, "running", "20261013T000000Z-beef"), "already running"),
            "worker down": (lambda w: setattr(w.worker, "down", True), "did not answer"),
            "resident paused": (lambda w: w.worker.residents_list[0].update(paused=True), "paused"),
            "no model assigned": (lambda w: w.config["garden"]["residents"].clear(), "no model assigned"),
            "model not in the table": (lambda w: w.assign("ada", "ghost"), "not in garden.models"),
            "layout unresolved": (lambda w: w.assign("ada", "broken"), "not known"),
            "switched off": (lambda w: w.config["garden"]["schedule"].update(enabled=False), "enabled is false"),
        }
        for name, (break_it, expect) in cases.items():
            with self.subTest(name):
                w = self.world()
                break_it(w)
                w.tick()
                self.assertEqual(w.lease.acquired, [], name)
                self.assertEqual(w.worker.started, [], name)
                self.assertIn(expect, w.sched.reason, name)

    def test_the_other_gpus_gate_does_not_matter_but_this_ones_does(self):
        w = self.world()
        w.close_gate("busy", gpu=OTHER)
        w.tick()
        self.assertEqual(len(w.lease.acquired), 1)

    def test_a_refused_acquire_is_not_retried_every_tick(self):
        w = self.world()
        w.lease.refuse = "in use: Computer answered 20 s ago (force to override)"
        w.tick()
        self.assertEqual(w.worker.started, [])
        self.assertIn("lease was refused", w.sched.reason)
        w.lease.refuse = None
        w.tick(1, step=20)
        self.assertEqual(w.lease.acquired, [])          # retry_s = 60
        w.clock.advance(61)
        w.tick()
        self.assertEqual(len(w.lease.acquired), 1)

    def test_a_failed_first_session_gives_the_lease_back(self):
        w = self.world()
        w.worker.refuse = True
        w.tick()
        self.assertEqual(len(w.lease.released), 1)
        self.assertIsNone(w.sched.window)
        self.assertIsNone(w.active.engine())

    def test_budgets_per_resident(self):
        w = self.world(max_sessions_24h=2)
        for i in range(2):
            w.sched.sessions.started("s%d" % i, "ada", "bonsai", "scheduler")
        w.tick()
        self.assertEqual(w.lease.acquired, [])
        self.assertIn("2 sessions in 24 h", w.sched.reason)
        w.clock.advance(86400 + 1)                       # rolling: they age out
        w.tick()
        self.assertEqual(len(w.lease.acquired), 1)
        w = self.world(max_tokens_hour=1000)
        w.sched.sessions.started("s0", "ada", "bonsai", "scheduler")
        w.sched.sessions.ended("s0", 1200, "rest")
        w.tick()
        self.assertEqual(w.lease.acquired, [])
        self.assertIn("tokens in the last hour", w.sched.reason)
        w.clock.advance(3601)
        w.tick()
        self.assertEqual(len(w.lease.acquired), 1)

    def test_a_token_cap_is_passed_once_the_speed_is_known(self):
        w = self.world()
        w.config["garden"]["models"]["bonsai"]["tok_s"] = 10
        w.tick()
        body = w.worker.started[0][0]
        self.assertAlmostEqual(body["max_tokens"], 28 * 60 * 10, delta=30)
        w = self.world()
        w.sched.sessions.started("old", "ada", "bonsai", "scheduler")
        w.clock.advance(100)
        w.sched.sessions.ended("old", 2000, "rest")       # 20 tok/s measured
        w.clock.advance(100000)
        w.tick()
        self.assertAlmostEqual(w.worker.started[0][0]["max_tokens"], 28 * 60 * 20, delta=60)

    def test_skips_are_traced_once_per_reason(self):
        w = self.world()
        w.pending = 1
        w.tick(5, step=20)
        self.assertEqual(len(w.events("skipped")), 1)
        w.pending = 2                                    # same reason once the number is ignored
        w.tick()
        self.assertEqual(len(w.events("skipped")), 1)
        w.pending, w.last_turn = 0, w.clock() - 60
        w.tick()
        self.assertEqual(len(w.events("skipped")), 2)
        w.clock.advance(1000)
        w.last_turn = w.clock() - 60
        w.tick()
        self.assertEqual(len(w.events("skipped")), 3)


# ---- the window: ttl, early release, round robin ---------------------------------------------------------------------------
class WindowTests(Base):
    def test_the_session_and_lease_end_twelve_minutes_before_the_leases_own_end(self):
        w = self.world()
        w.tick()
        lease = w.lease.lease
        w.clock.t = lease["expires"] - 12 * 60 - 1
        w.tick()
        self.assertIsNotNone(w.sched.window)
        w.clock.advance(2)
        w.tick()
        self.assertIsNone(w.sched.window)
        rel = w.lease.released[0]
        self.assertGreaterEqual(rel["expires"] - rel["at"], 12 * 60 - 5)         # the check-in is at 10: it never fires
        self.assertEqual(len(w.worker.stopped), 1)
        self.assertIn("garden scheduler", w.worker.stopped[0][1])
        self.assertEqual(w.events("released")[0]["outcome"], "released")
        self.assertIsNone(w.active.engine())

    def test_the_floor_cannot_be_configured_away(self):
        w = self.world(stop_before_end_min=2)
        w.tick()
        self.assertEqual(w.lease.lease["expires"] - w.sched.window["deadline"], 12 * 60)

    def test_ttl_is_the_estimate_plus_the_margin_and_a_lease_is_never_renewed(self):
        w = self.world(default_window_min=20, margin_min=10)
        w.tick()
        self.assertEqual(w.lease.acquired[0]["ttl_s"], 30 * 60)
        self.assertFalse(hasattr(w.lease, "renew"))

    def test_sessions_run_one_after_another_round_robin_while_time_remains(self):
        w = self.world()
        w.worker.residents_list = [{"id": "ada", "paused": False}, {"id": "bo", "paused": False}, {"id": "cy", "paused": False}]
        w.assign("bo", "bonsai")
        w.assign("cy", "big")                           # another model: not in this window
        w.tick()
        order = [w.worker.started[0][0]["resident"]]
        for _ in range(3):
            w.worker.finish()
            w.clock.advance(60)
            w.tick()
            order.append(w.worker.started[-1][0]["resident"])
        self.assertEqual(order, ["ada", "bo", "ada", "bo"])
        self.assertEqual(len(w.lease.acquired), 1)       # one lease for the whole window

    def test_the_first_resident_rotates_between_windows(self):
        w = self.world()
        w.worker.residents_list = [{"id": "ada", "paused": False}, {"id": "bo", "paused": False}]
        w.assign("bo", "bonsai")
        firsts = []
        for _ in range(2):
            w.tick()
            firsts.append(w.worker.started[-1][0]["resident"])
            w.sched._end_window("test", preempted=False, own=False)
            w.clock.advance(400)
        self.assertEqual(firsts, ["ada", "bo"])

    def test_no_session_starts_with_less_than_min_session_time_left(self):
        w = self.world()
        w.tick()
        w.clock.t = w.sched.window["deadline"] - 100
        w.worker.finish()
        w.tick()
        self.assertEqual(len(w.worker.started), 1)       # 100 s < 180 s: the window closes instead
        self.assertIsNone(w.sched.window)
        self.assertEqual(len(w.lease.released), 1)

    def test_our_own_end_is_marked_so_the_following_close_is_not_a_sample(self):
        w = self.world()
        w.tick()
        w.clock.t = w.sched.window["deadline"] + 1
        w.tick()
        w.close_gate("GPU 60 % busy")
        w.clock.advance(120)
        w.tick()
        closed = w.sched.history.closed[GPU][-1]
        self.assertTrue(closed["own"])
        self.assertEqual(w.sched.history.lengths(GPU), [])

    def test_after_a_window_nothing_starts_for_the_cooldown(self):
        w = self.world()
        w.tick()
        w.sched._end_window("test", preempted=False, own=False)
        w.tick()
        self.assertEqual(len(w.lease.acquired), 1)
        w.clock.advance(301)
        w.tick()
        self.assertEqual(len(w.lease.acquired), 2)


# ---- preemption --------------------------------------------------------------------------------------------------------------
class PreemptionTests(Base):
    def running(self, **cfg):
        w = self.world(**cfg)
        w.tick()
        self.assertEqual(w.sched.state, "running")
        w.clock.advance(5)
        return w

    def assert_preempted(self, w, expect):
        self.assertIsNone(w.sched.window)
        self.assertEqual(w.sched.state, "preempted")
        self.assertEqual(len(w.worker.stopped), 1, "the session was not stopped")
        self.assertEqual(len(w.lease.released), 1, "the lease was not released")
        self.assertIsNone(w.lease.lease)
        self.assertIsNone(w.active.engine())
        ev = w.events("preempted")[0]
        self.assertIn(expect, ev["why"])
        self.assertEqual(ev["outcome"], "preempted")

    def test_the_gate_closing(self):
        w = self.running()
        w.gate["shared_reasons"] = ["A resident is awake (phone used 1 min ago)"]
        w.tick()
        self.assert_preempted(w, "A resident is awake")

    def test_an_approval_appearing(self):
        w = self.running()
        w.pending = 1
        w.tick()
        self.assert_preempted(w, "approval")

    def test_computer_getting_a_turn(self):
        w = self.running()
        w.last_turn = w.clock() - 1
        w.tick()
        self.assert_preempted(w, "Computer got a turn")

    def test_the_cards_own_busy_reasons_do_not_preempt_our_own_lease(self):
        w = self.running()
        w.close_gate("GPU 80 % busy", "vision: no metrics")      # our session is what makes it busy
        w.tick()
        self.assertEqual(w.sched.state, "running")

    def test_preemption_is_checked_every_poll_even_with_the_gate_cached(self):
        w = self.running()
        w.pending = 3
        w.tick()                                           # one tick is enough: no waiting for the next gate read
        self.assertIsNone(w.sched.window)

    def test_the_lease_vanishing_stops_the_session_without_a_second_release(self):
        w = self.running()
        w.lease.lease = None                              # the phone's Stop, or an expiry
        w.tick()
        self.assertIsNone(w.sched.window)
        self.assertEqual(len(w.worker.stopped), 1)
        self.assertEqual(w.lease.released, [])

    def test_a_worker_that_stops_answering_ends_the_window(self):
        w = self.running()
        w.worker.down = True
        w.tick()
        self.assertIsNone(w.sched.window)
        self.assertEqual(len(w.lease.released), 1)

    def test_switching_the_schedule_off_ends_the_window_without_a_restart(self):
        w = self.running()
        w.config["garden"]["schedule"]["enabled"] = False
        w.tick()
        self.assertIsNone(w.sched.window)
        self.assertEqual(len(w.lease.released), 1)
        self.assertEqual(w.sched.state, "waiting")
        w.tick()
        self.assertEqual(w.sched.state, "disabled")

    def test_after_a_preemption_it_waits_out_the_cooldown_before_trying_again(self):
        w = self.running()
        w.pending = 1
        w.tick()
        w.pending = 0
        w.tick(3, step=20)
        self.assertEqual(len(w.lease.acquired), 1)
        self.assertEqual(w.sched.state, "preempted")
        w.clock.advance(400)
        w.tick()
        self.assertEqual(len(w.lease.acquired), 2)

    def test_a_failed_release_is_reported_not_hidden(self):
        w = self.running()
        w.lease.release_ok = False
        w.pending = 1
        w.tick()
        self.assertEqual(w.events("released")[0]["outcome"], "failed")


# ---- layouts that evict the coordinator -----------------------------------------------------------------------------------
class CoordinatorGuardTests(Base):
    def test_never_by_default(self):
        for model, layout in (("big", "longctx_6750"),):
            w = self.world()
            w.assign("ada", model)
            w.tick()
            self.assertEqual(w.lease.acquired, [], layout)
            self.assertIn("coordinator", w.sched.reason)
        w = self.world()
        w.config["garden"]["models"]["max"] = {"layout": "longctx_max", "engine_url": "http://engine:8005/v1"}
        w.assign("ada", "max")
        w.tick()
        self.assertEqual(w.lease.acquired, [])

    def test_only_when_the_flag_is_set(self):
        w = self.world(allow_coordinator_layouts=True)
        w.assign("ada", "big")
        w.tick()
        self.assertEqual([a["layout"] for a in w.lease.acquired], ["longctx_6750"])

    def test_the_guard_does_not_depend_on_the_layouts_name(self):
        w = self.world()
        w.config["garden"]["models"]["innocent"] = {"layout": "friendly_name", "engine_url": "http://engine:8005/v1"}
        LAYOUTS["friendly_name"] = {"layout": "friendly_name", "roles": ["coordinator"], "gpus": [GPU], "resolved": True, "disabled": None}
        self.addCleanup(LAYOUTS.pop, "friendly_name", None)
        w.assign("ada", "innocent")
        w.tick()
        self.assertEqual(w.lease.acquired, [])

    def test_a_disabled_layout_is_refused(self):
        w = self.world()
        LAYOUTS["longctx_6600"]["disabled"] = "crashed at 245K"
        self.addCleanup(LAYOUTS["longctx_6600"].update, disabled=None)
        w.tick()
        self.assertEqual(w.lease.acquired, [])
        self.assertIn("disabled", w.sched.reason)


# ---- the CPU slot ----------------------------------------------------------------------------------------------------------
class CpuSlotTests(Base):
    def test_off_by_default(self):
        w = self.world()
        w.assign("ada", "granite-cpu")
        w.tick()
        self.assertEqual(w.units, [])
        self.assertIn("CPU slot is switched off", w.sched.reason)

    def test_when_on_it_starts_the_unit_runs_a_session_and_stops_the_unit(self):
        w = self.world(cpu_enabled=True)
        w.assign("ada", "granite-cpu")
        w.tick()
        self.assertEqual(w.units, [("start", "llama-courage-lite.service")])
        self.assertEqual(w.lease.acquired, [])
        self.assertEqual(w.worker.started[0][0]["model"], "granite-cpu")
        w.clock.t = w.sched.window["deadline"] + 1
        w.tick()
        self.assertEqual(w.units[-1], ("stop", "llama-courage-lite.service"))
        self.assertIsNone(w.sched.window)

    def test_a_lease_appearing_wins_at_once_and_the_unit_is_left_alone(self):
        w = self.world(cpu_enabled=True)
        w.assign("ada", "granite-cpu")
        w.tick()
        w.clock.advance(5)
        w.lease.lease = {"id": "other", "layout": "longctx_6600", "expires": w.clock() + 3600, "state": "active"}
        w.tick()
        self.assertIsNone(w.sched.window)
        self.assertEqual(len(w.worker.stopped), 1)
        self.assertEqual(w.units, [("start", "llama-courage-lite.service")])      # the lease may be using it as the stand-in
        self.assertEqual(w.events("preempted")[0]["why"][:20], "an engine lease took")

    def test_the_house_gate_applies_and_a_unit_that_never_answers_is_stopped(self):
        w = self.world(cpu_enabled=True)
        w.assign("ada", "granite-cpu")
        w.gate["shared_reasons"] = ["A resident is awake"]
        w.tick()
        self.assertEqual(w.units, [])
        w = self.world(cpu_enabled=True, cpu_wait_s=10)
        w.assign("ada", "granite-cpu")
        w.health_ok = False
        w.tick()
        self.assertEqual(w.units, [("start", "llama-courage-lite.service"), ("stop", "llama-courage-lite.service")])
        self.assertIsNone(w.sched.window)


# ---- manual vs scheduled -----------------------------------------------------------------------------------------------------
class FakeH:
    def __init__(self, principal=auth.OWNER, command="GET"):
        self.principal, self.command, self._raw_body, self.wfile = principal, command, b"", io.BytesIO()
        self.status = None

    def send_response(self, s):
        self.status = s

    def send_header(self, *a):
        pass

    def end_headers(self):
        pass

    def answer(self):
        return json.loads(self.wfile.getvalue() or b"{}")


class RouteTests(Base):
    def setUp(self):
        self.w = self.world()
        self.saved = []
        self.disk = {"garden": {"worker_url": "http://w", "control_key": "k" * 20}}
        self.disk["garden"].update(json.loads(json.dumps(self.w.config["garden"])))
        self.ctx = apps.AppContext(self.w.config, ROOT, lambda: self.w.dir, services={
            "load_config": lambda: json.loads(json.dumps(self.disk)), "save_config": lambda cfg: (self.saved.append(cfg), self.disk.update(cfg))})
        self.client = FakeClient(self.w.worker)
        garden_client.install(self.ctx, client=self.client, feeder=type("F", (), {"status": {}, "run_once": lambda s: {}, "start": lambda s: None})(),
                              scheduler=self.w.sched)

    def call(self, method, path, body=None, principal=auth.OWNER):
        parsed = urllib.parse.urlparse(path)
        fn = self.ctx.routes.find(method, parsed.path)
        self.assertIsNotNone(fn, path)
        h = FakeH(principal, method)
        fn(h, parsed, body or {})
        return h.status, h.answer()

    def test_a_manual_start_is_refused_while_the_scheduler_holds_a_window(self):
        self.w.tick()
        status, out = self.call("POST", "/api/garden/sessions/start", {"resident": "ada"})
        self.assertEqual(status, 409)
        self.assertIn("scheduler holds", out["error"])
        self.assertIn("longctx_6600", out["error"])
        self.assertEqual(len(self.w.worker.started), 1)                # only the scheduler's

    def test_a_manual_start_works_otherwise_and_the_scheduler_then_waits(self):
        self.w.config["garden"]["schedule"]["enabled"] = True
        status, out = self.call("POST", "/api/garden/sessions/start", {"resident": "ada"})
        self.assertEqual(status, 200)
        self.assertEqual(self.client.started[-1][1], "manual")
        self.w.tick()
        self.assertEqual(self.w.lease.acquired, [])
        self.assertIn("already running", self.w.sched.reason)
        self.assertEqual(self.w.sched.sessions.count_24h("ada"), 1)       # counted against the budget

    def test_the_schedule_status_and_toggle_persist(self):
        status, out = self.call("GET", "/api/garden/schedule")
        self.assertEqual((status, out["enabled"], out["state"]), (200, True, "disabled"))
        status, out = self.call("POST", "/api/garden/schedule", {"enabled": False})
        self.assertEqual((status, out["enabled"], out["saved"]), (200, False, True))
        self.assertFalse(self.w.config["garden"]["schedule"]["enabled"])           # live
        self.assertFalse(self.disk["garden"]["schedule"]["enabled"])               # on disk
        self.assertEqual(self.call("POST", "/api/garden/schedule", {"enabled": "yes"})[0], 400)
        self.assertEqual(self.call("POST", "/api/garden/schedule", {"enabled": True}, principal=garden_client.SCOPED)[0], 401)
        self.assertEqual(self.call("GET", "/api/garden/schedule", principal=None)[0], 401)

    def test_without_a_way_to_save_the_toggle_is_memory_only_and_says_so(self):
        ctx = apps.AppContext(self.w.config, ROOT, lambda: self.w.dir)
        garden_client.install(ctx, client=self.client, feeder=type("F", (), {"status": {}, "start": lambda s: None})(), scheduler=self.w.sched)
        fn = ctx.routes.find("POST", "/api/garden/schedule")
        h = FakeH(auth.OWNER, "POST")
        fn(h, urllib.parse.urlparse("/api/garden/schedule"), {"enabled": False})
        out = h.answer()
        self.assertEqual((out["saved"], self.w.config["garden"]["schedule"]["enabled"]), (False, False))
        self.assertIn("memory", out["note"])

    def test_the_model_select_assigns_and_validates(self):
        status, out = self.call("POST", "/api/garden/residents/ada/model", {"model": "big"})
        self.assertEqual((status, out["model"], out["saved"]), (200, "big", True))
        self.assertEqual(self.w.config["garden"]["residents"]["ada"]["model"], "big")
        self.assertEqual(self.disk["garden"]["residents"]["ada"]["model"], "big")
        self.assertEqual(self.call("POST", "/api/garden/residents/ada/model", {"model": "ghost"})[0], 400)
        self.assertEqual(self.w.config["garden"]["residents"]["ada"]["model"], "big")
        self.call("POST", "/api/garden/residents/ada/model", {"model": ""})
        self.assertNotIn("model", self.w.config["garden"]["residents"]["ada"])
        self.assertEqual(self.call("POST", "/api/garden/residents/Bad Id/model", {"model": "big"})[0], 404)
        self.assertEqual(self.call("POST", "/api/garden/residents/ada/model", {"model": "big"}, principal=garden_client.SCOPED)[0], 401)
        self.assertEqual(self.w.sched.status()["assignments"], {"ada": ""})

    def test_no_scheduler_means_a_clear_503(self):
        ctx = apps.AppContext({"garden": {}}, ROOT, lambda: self.w.dir)
        garden_client.install(ctx, client=self.client, feeder=type("F", (), {"status": {}, "start": lambda s: None})())
        fn = ctx.routes.find("GET", "/api/garden/schedule")
        h = FakeH(auth.OWNER, "GET")
        fn(h, urllib.parse.urlparse("/api/garden/schedule"), {})
        self.assertEqual(h.status, 503)
        self.assertIsNone(garden_client.SCHEDULER)
        garden_client.SCHEDULER = self.w.sched


class FakeClient:
    """The slice of GardenClient the routes use, over the fake worker."""

    def __init__(self, worker):
        self.worker, self.active, self.started = worker, worker.active, []

    def start_session(self, body, by="manual"):
        body = dict({"model": "bonsai"}, **body)
        out = self.worker.start_session(body, by)
        self.started.append((body, by))
        return out


# ---- status and metrics ------------------------------------------------------------------------------------------------------
class ReportTests(Base):
    def test_status_shape(self):
        w = self.world()
        w.tick()
        st = w.sched.status()
        self.assertEqual((st["enabled"], st["state"]), (True, "running"))
        self.assertTrue(st["gates"][GPU]["open"])
        self.assertEqual(st["estimates"][GPU]["basis"], "default")
        self.assertEqual(st["window"]["layout"], "longctx_6600")
        self.assertEqual(st["sessions_24h"], {"ada": 1})
        self.assertEqual(set(st["models"]), {"bonsai", "big", "granite-cpu", "broken"})
        self.assertEqual(st["last_event"]["event"], "scheduled_start")

    def test_metrics(self):
        w = self.world()
        w.tick()
        lines = w.sched.metrics()
        self.assertIn('stonesage_garden_scheduler_state{state="running"} 1', lines)
        self.assertIn('stonesage_garden_scheduler_state{state="idle"} 0', lines)
        self.assertIn("stonesage_garden_scheduler_enabled 1", lines)
        self.assertIn('stonesage_garden_sessions_24h{resident="ada"} 1', lines)


# ---- the proxy follows the running session -----------------------------------------------------------------------------------
class FakeResp:
    def __init__(self, data):
        self._d = json.dumps(data).encode()

    def read(self, *a):
        return self._d

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class ProxyEngineTests(Base):
    def setUp(self):
        self.seen = []
        self.config = {"garden": {"engine_url": "http://default:8009/v1", "worker_url": "http://w", "control_key": "k" * 20,
                                  "models": {"bonsai": {"layout": "longctx_6600", "engine_url": "http://engine:8005/v1", "model_sha256": "a" * 64},
                                             "granite-cpu": {"layout": None, "engine_url": "http://engine:8006/v1", "model_sha256": "b" * 64},
                                             "plain": {"layout": None}},
                                  "residents": {"ada": {"model": "bonsai"}, "bo": {"model": "granite-cpu"}, "cy": {"model": "plain"}}}}
        self.posts = []

        def opener(req, timeout=None):
            if req.full_url.endswith("/sessions/start"):
                self.posts.append(json.loads(req.data))
                if self.refuse:
                    import urllib.error
                    raise urllib.error.HTTPError(req.full_url, 409, "busy", {}, io.BytesIO(b'{"error": "a session is already running"}'))
                return FakeResp({"ok": True, "session": "20261013T000001Z-aaaa", "resident": self.posts[-1]["resident"]})
            self.seen.append(req.full_url)
            return FakeResp({"choices": [{"message": {"content": "hi"}}], "usage": {}})

        self.refuse = False
        self.client = garden_client.GardenClient(self.config, opener=opener)
        self.route = garden_client.model_proxy(self.config, opener, engine_of=self.client.active.engine)

    def chat(self):
        h = FakeH(garden_client.SCOPED, "POST")
        self.route(h, urllib.parse.urlparse("/api/garden/v1/chat/completions"), {"messages": [{"role": "user", "content": "x"}]})
        return h.status

    def test_each_session_goes_to_its_own_models_engine_with_no_config_edit(self):
        self.assertEqual(self.chat(), 200)
        self.assertEqual(self.seen[-1], "http://default:8009/v1/chat/completions")          # nothing running: the fallback
        self.client.start_session({"resident": "ada"})
        self.assertEqual(self.posts[-1]["model"], "bonsai")
        self.assertEqual(self.posts[-1]["model_sha256"], "a" * 64)
        self.chat()
        self.assertEqual(self.seen[-1], "http://engine:8005/v1/chat/completions")
        self.client.stop_session("20261013T000001Z-aaaa")
        self.client.start_session({"resident": "bo"})
        self.assertEqual(self.posts[-1]["model"], "granite-cpu")
        self.chat()
        self.assertEqual(self.seen[-1], "http://engine:8006/v1/chat/completions")
        self.client.active.clear("20261013T000001Z-aaaa")
        self.chat()
        self.assertEqual(self.seen[-1], "http://default:8009/v1/chat/completions")

    def test_an_explicit_model_beats_the_assignment_and_unknown_names_keep_the_old_setup(self):
        self.client.start_session({"resident": "ada", "model": "granite-cpu"})
        self.assertEqual(self.posts[-1]["model"], "granite-cpu")
        self.chat()
        self.assertEqual(self.seen[-1], "http://engine:8006/v1/chat/completions")
        self.client.active.clear()
        self.config["garden"]["model"] = "legacy-name"
        self.config["garden"]["model_sha256"] = "c" * 64
        self.client.start_session({"resident": "nobody"})
        self.assertEqual((self.posts[-1]["model"], self.posts[-1]["model_sha256"]), ("legacy-name", "c" * 64))
        self.chat()
        self.assertEqual(self.seen[-1], "http://default:8009/v1/chat/completions")

    def test_a_model_without_an_engine_url_falls_back_to_the_default_engine(self):
        self.client.start_session({"resident": "cy"})
        self.chat()
        self.assertEqual(self.seen[-1], "http://default:8009/v1/chat/completions")

    def test_a_refused_start_gives_the_running_sessions_engine_back(self):
        self.client.start_session({"resident": "ada"})
        self.refuse = True
        with self.assertRaises(GardenError):
            self.client.start_session({"resident": "bo"})
        self.chat()
        self.assertEqual(self.seen[-1], "http://engine:8005/v1/chat/completions")          # still bonsai's session

    def test_the_active_engine_expires_by_itself(self):
        clock = Clock()
        a = ActiveEngine(clock)
        a.set("http://e/v1", "m", "scheduler", ttl_s=100)
        self.assertEqual(a.engine(), "http://e/v1")
        clock.advance(100 + garden_client.ACTIVE_SLACK_S + 1)
        self.assertIsNone(a.engine())

    def test_clear_only_clears_its_own_session(self):
        a = ActiveEngine()
        a.set("http://e/v1", "m", "manual")
        a.bind("S1")
        a.clear("S2")
        self.assertEqual(a.engine(), "http://e/v1")
        a.clear("S1")
        self.assertIsNone(a.engine())


if __name__ == "__main__":
    unittest.main()
