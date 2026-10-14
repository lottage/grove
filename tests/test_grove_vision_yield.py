"""Vision during garden time, the garden scheduler's side (2026-10-14; docs/garden/DESIGN.md "Vision during garden time").

When Computer's camera or vision work takes the RX 6600 back from the garden, the scheduler stops a 6600 session, starts only the
vision cooldown, and never holds back the coordinator layouts for it. Grove's tree only: the fakes are test_grove_scheduler.py's.
The platform side (the queue, the hot load, the takeback) is tests/test_vision_yield.py.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_grove_scheduler as tgs  # noqa: E402
from grove.stonesage_app import scheduler as gs  # noqa: E402,F401  (the grove layer: this file belongs to the grove tree only)


# ---- the garden scheduler's side ---------------------------------------------------------------------------------------------------
class TestSchedulerCards(tgs.Base):
    def world(self, **cfg):
        w = tgs.World(self, **cfg)
        w.cooldowns = {"coordinator": 0.0, "vision": 0.0}
        w.sched.takeback_cooldown_fn = lambda role="coordinator": w.cooldowns[role]
        return w

    def start(self, model, roles, **cfg):
        w = self.world(allow_coordinator_layouts=True, **cfg)
        w.assign("ada", model)
        w.tick()
        self.assertEqual(len(w.lease.acquired), 1)
        w.lease.lease["evicted_roles"] = roles
        return w

    def test_a_vision_takeback_stops_a_6600_session_and_starts_only_the_vision_cooldown(self):
        w = self.start("bonsai", ["vision"])
        sid = w.worker.running
        out = w.sched.yield_to_computer("takeback: tool", 600, resource="vision")
        self.assertTrue(out["stopped"])
        self.assertEqual(w.worker.stopped, [(sid, "takeback: tool")])
        self.assertGreater(w.sched.takeback_left("vision"), 500)
        self.assertEqual(w.sched.takeback_left("coordinator"), 0)

    def test_a_vision_takeback_leaves_a_6750_session_alone(self):
        w = self.start("big", ["coordinator", "worker"])
        out = w.sched.yield_to_computer("takeback: tool", 600, resource="vision")
        self.assertFalse(out["stopped"])
        self.assertEqual(w.worker.stopped, [])
        self.assertIsNotNone(w.sched.window)
        self.assertEqual(w.sched.takeback_left("vision"), 0)                # nothing was taken, so nothing cools down

    def test_a_coordinator_takeback_leaves_a_6600_session_alone(self):
        w = self.start("bonsai", ["vision"])
        self.assertFalse(w.sched.yield_to_computer("takeback: cockpit", 3600)["stopped"])
        self.assertEqual(w.worker.stopped, [])

    def test_no_6600_layout_for_ten_minutes_then_again_while_the_6750_is_not_held_back(self):
        w = self.start("bonsai", ["vision"])
        w.sched.yield_to_computer("takeback: sweep", 600, resource="vision")
        w.tick()
        w.lease.lease = None
        w.clock.advance(400)                                                # past the ordinary window cooldown (300 s)
        w.tick()
        self.assertEqual(len(w.lease.acquired), 1)
        self.assertIn("took the camera engine back", w.sched.reason)
        w.clock.advance(300)
        w.tick()
        self.assertEqual(len(w.lease.acquired), 2)                          # ten minutes are over
        w2 = self.world(allow_coordinator_layouts=True)
        w2.assign("ada", "big")
        w2.cooldowns["vision"] = 590
        w2.tick()
        self.assertEqual([a["layout"] for a in w2.lease.acquired], ["longctx_6750"])

    def test_the_platforms_per_card_cooldown_counts_and_an_old_zero_argument_function_is_the_coordinators(self):
        w = self.world()
        w.cooldowns["vision"] = 900
        w.tick()
        self.assertEqual(w.lease.acquired, [])
        self.assertIn("camera engine", w.sched.reason)
        w2 = tgs.World(self)
        w2.sched.takeback_cooldown_fn = lambda: 1800                        # written before cards: it speaks for the coordinator only
        w2.tick()
        self.assertEqual([a["layout"] for a in w2.lease.acquired], ["longctx_6600"])

    def test_status_shows_both_blocks(self):
        w = self.world()
        w.cooldowns.update(coordinator=120, vision=45)
        st = w.sched.status()["takeback"]
        self.assertEqual((st["coordinator_layouts_blocked_for_s"], st["vision_layouts_blocked_for_s"]), (120, 45))


if __name__ == "__main__":
    unittest.main()
