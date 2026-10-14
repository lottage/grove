"""The garden scheduler's side of "Computer takes the GPU back" (2026-10-13): coordinator layouts only with the flag, a passed stand-in
eval and no takeback cooldown; models with several layouts; the takeback hook; the CPU slot never running while Computer is on the stand-in;
a person at the cockpit keeping the garden off. No LAN: the fakes of test_grove_scheduler.py."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_grove_scheduler as tgs  # noqa: E402



# ---- the garden scheduler's side ---------------------------------------------------------------------------------------------
class SchedulerBase(tgs.Base):
    def world(self, **cfg):
        w = tgs.World(self, **cfg)
        w.cooldown = [0.0]
        w.sched.takeback_cooldown_fn = lambda: w.cooldown[0]
        return w


class TestSchedulerTakeback(SchedulerBase):
    def start_big(self, **cfg):
        w = self.world(allow_coordinator_layouts=True, **cfg)
        w.assign("ada", "big")
        w.tick()
        self.assertEqual([a["layout"] for a in w.lease.acquired], ["longctx_6750"])
        return w

    def test_a_coordinator_layout_needs_the_flag_and_a_ready_stand_in(self):
        w = self.world()
        w.assign("ada", "big")
        w.tick()
        self.assertEqual(w.lease.acquired, [])
        self.assertIn("allow_coordinator_layouts is false", w.sched.reason)
        w = self.world(allow_coordinator_layouts=True)
        w.assign("ada", "big")
        tgs.LAYOUTS["longctx_6750"]["stand_in_ready"] = False
        self.addCleanup(tgs.LAYOUTS["longctx_6750"].update, stand_in_ready=True)
        w.tick()
        self.assertEqual(w.lease.acquired, [])
        self.assertIn("stand-in is not ready", w.sched.reason)
        tgs.LAYOUTS["longctx_6750"].pop("stand_in_ready")                    # a layout record that does not say at all
        w.tick(1, 60)
        self.assertEqual(w.lease.acquired, [])

    def test_with_both_the_garden_borrows_the_coordinator_and_marks_the_lease_background_work(self):
        w = self.start_big()
        self.assertTrue(w.lease.acquired[0]["yield_to_computer"])

    def test_a_layout_that_leaves_the_coordinator_alone_is_not_held_to_the_stand_in_rule(self):
        w = self.world()
        w.tick()
        self.assertEqual([a["layout"] for a in w.lease.acquired], ["longctx_6600"])
        self.assertTrue(w.lease.acquired[0]["yield_to_computer"])

    def test_takeback_stops_the_session_with_its_reason_ends_the_window_and_starts_the_cooldown(self):
        w = self.start_big()
        sid = w.worker.running
        out = w.sched.yield_to_computer("takeback: tool", 3600)
        self.assertTrue(out["stopped"])
        self.assertEqual(w.worker.stopped, [(sid, "takeback: tool")])            # the worker keeps the draft with this reason
        w.tick()
        self.assertIsNone(w.sched.window)
        self.assertEqual(w.lease.released, [])                                   # the takeback releases the lease, not the scheduler
        self.assertEqual(len(w.worker.stopped), 1)                               # and the session was stopped once
        pre = [e for e in w.events("preempted")]
        self.assertEqual(pre[0]["why"], "takeback: tool")
        self.assertEqual(w.sched.state, "preempted")
        self.assertEqual(w.sched.sessions.rows[sid]["reason"], "preempted: takeback: tool")

    def test_no_coordinator_layout_for_the_cooldown_then_again(self):
        w = self.start_big()
        w.sched.yield_to_computer("takeback: cockpit", 3600)
        w.tick()
        w.lease.lease = None
        w.clock.advance(400)                                                     # past the ordinary window cooldown (300 s)
        w.tick()
        self.assertEqual(len(w.lease.acquired), 1)
        self.assertIn("took the coordinator back", w.sched.reason)
        self.assertGreater(w.sched.takeback_left(), 3000)
        w.clock.advance(3600)
        w.tick()
        self.assertEqual([a["layout"] for a in w.lease.acquired], ["longctx_6750", "longctx_6750"])

    def test_the_platform_s_cooldown_counts_too_and_survives_a_scheduler_restart(self):
        w = self.world(allow_coordinator_layouts=True)
        w.assign("ada", "big")
        w.cooldown[0] = 1800
        w.tick()
        self.assertEqual(w.lease.acquired, [])
        self.assertIn("not again for 30 min", w.sched.reason)
        w.cooldown[0] = 0
        w.tick(1, 60)
        self.assertEqual(len(w.lease.acquired), 1)

    def test_the_cooldown_does_not_hold_back_a_layout_that_leaves_the_coordinator_alone(self):
        w = self.world(allow_coordinator_layouts=True)
        w.cooldown[0] = 3000
        w.tick()
        self.assertEqual([a["layout"] for a in w.lease.acquired], ["longctx_6600"])

    def test_a_takeback_with_no_window_open_does_nothing_but_start_the_cooldown(self):
        w = self.world(allow_coordinator_layouts=True)
        self.assertEqual(w.sched.yield_to_computer("takeback: cockpit", 600)["stopped"], False)
        w.tick()
        self.assertIsNone(w.sched.yield_request)
        self.assertGreater(w.sched.takeback_left(), 500)

    def test_a_stale_takeback_does_not_end_the_next_window(self):
        w = self.world()
        w.sched.yield_request = {"why": "takeback: tool", "at": 0}
        w.tick()                                                                 # _maybe_start clears it, then starts a window
        self.assertIsNotNone(w.sched.window)
        w.tick()
        self.assertIsNotNone(w.sched.window)

    def test_status_shows_the_last_takeback(self):
        w = self.world(allow_coordinator_layouts=True)
        w.sched.takeback_status_fn = lambda: {"last": {"why": "tool", "outcome": "ok"}}
        w.cooldown[0] = 120
        st = w.sched.status()["takeback"]
        self.assertEqual(st["last"]["why"], "tool")
        self.assertEqual(st["coordinator_layouts_blocked_for_s"], 120)

    # the cockpit signal, through the gate
    def test_the_garden_does_not_start_while_a_person_is_on_the_page(self):
        w = self.world()
        w.gate["shared_reasons"] = ["someone is using StoneSage (12 s ago)"]
        for g in w.gate["gpus"].values():
            g["open"] = False
        w.tick()
        self.assertEqual(w.lease.acquired, [])
        self.assertIn("someone is using StoneSage", w.sched.reason)

    def test_a_running_window_ends_when_a_person_arrives(self):
        w = self.world()
        w.tick()
        self.assertIsNotNone(w.sched.window)
        w.gate["shared_reasons"] = ["someone is using StoneSage (2 s ago)"]
        w.clock.advance(30)
        w.tick()
        self.assertIsNone(w.sched.window)
        self.assertIn("someone is using StoneSage", w.events("preempted")[0]["why"])
        self.assertEqual(len(w.lease.released), 1)

    # models with several layouts
    def multi(self, **cfg):
        w = self.world(**cfg)
        w.config["garden"]["models"]["bonsai-any"] = {"layouts": ["longctx_6600", "longctx_6750"], "engine_url": "http://engine:8005/v1"}
        w.assign("ada", "bonsai-any")
        return w

    def test_the_6750_is_preferred_when_coordinator_layouts_are_allowed(self):
        w = self.multi(allow_coordinator_layouts=True)
        w.tick()
        self.assertEqual([a["layout"] for a in w.lease.acquired], ["longctx_6750"])

    def test_the_6600_is_used_without_the_flag(self):
        w = self.multi()
        w.tick()
        self.assertEqual([a["layout"] for a in w.lease.acquired], ["longctx_6600"])

    def test_the_6750_is_used_when_only_its_gate_is_open(self):
        w = self.multi(allow_coordinator_layouts=True)
        w.close_gate("vision served a request in the last 5 min")
        w.tick()
        self.assertEqual([a["layout"] for a in w.lease.acquired], ["longctx_6750"])

    def test_the_6750_alone_is_not_used_without_the_flag(self):
        w = self.multi()
        w.close_gate("vision served a request in the last 5 min")
        w.tick()
        self.assertEqual(w.lease.acquired, [])
        self.assertIn("allow_coordinator_layouts is false", w.sched.reason)
        self.assertIn("idle gate is closed", w.sched.reason)

    def test_the_config_order_is_the_preference(self):
        w = self.multi(allow_coordinator_layouts=True)
        w.config["garden"]["models"]["bonsai-any"]["layouts"] = ["longctx_6750", "longctx_6600"]
        w.tick()
        self.assertEqual([a["layout"] for a in w.lease.acquired], ["longctx_6750"])

    def test_the_catalogue_names_every_slot(self):
        from grove.stonesage_app import models as gm
        text = gm.catalogue_text(self.multi().config, None, None)
        self.assertIn("longctx_6600 or longctx_6750", text)

    # the CPU slot and the stand-in are the same engine
    def test_the_cpu_slot_never_starts_while_computer_is_on_the_stand_in(self):
        w = self.world(cpu_enabled=True)
        w.assign("ada", "granite-cpu")
        w.lease.brain = lambda role: {"mode": "lite", "url": "http://engine:8006", "until": 1}
        w.tick()
        self.assertEqual(w.units, [])
        self.assertIn("spare brain", w.sched.reason)

    def test_the_cpu_slot_never_starts_while_a_lease_is_held(self):
        w = self.world(cpu_enabled=True)
        w.assign("ada", "granite-cpu")
        w.lease.lease = {"id": "L9", "layout": "longctx_6750", "expires": w.clock() + 3600, "state": "active"}
        w.tick()
        self.assertEqual(w.units, [])

    def test_a_running_cpu_window_ends_the_moment_a_lease_appears_and_leaves_the_unit_to_the_lease(self):
        w = self.world(cpu_enabled=True)
        w.assign("ada", "granite-cpu")
        w.tick()
        self.assertEqual(w.sched.window["kind"], "cpu")
        w.lease.lease = {"id": "L9", "layout": "longctx_6750", "expires": w.clock() + 3600, "state": "active"}
        w.clock.advance(6)
        w.tick()
        self.assertIsNone(w.sched.window)
        self.assertNotIn(("stop", "llama-courage-lite.service"), w.units)       # the lease may be using it as the stand-in
        self.assertIn("lease", w.events("preempted")[0]["why"])


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
