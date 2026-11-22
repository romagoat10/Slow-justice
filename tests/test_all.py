"""Unit tests. Run with either:
    python -m pytest tests
    python tests/test_all.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ctb.agents import Notes, Period, SyntheticAgent, parse_reply, render_prompt
from ctb.market import best_response, joint_monopoly_price, nash_price, profit
from ctb.measures import Fork, pi_dynamic, pi_one_period, pi_static
from ctb.scripted import PAPER_TABLE2, make_fork, table2
from ctb.stats import sign_test_p
from ctb.stickiness import ols, pooled_rho


class Market(unittest.TestCase):
    def test_constants(self):
        self.assertAlmostEqual(nash_price(), 1.473, places=3)
        self.assertAlmostEqual(joint_monopoly_price(), 1.925, places=3)

    def test_br_is_argmax(self):
        for q in (1.5, 1.9, 2.2):
            br = best_response(q)
            grid = np.linspace(1.01, 3, 4000)
            self.assertAlmostEqual(br, grid[np.argmax([profit(p, q) for p in grid])], places=2)


class Measures(unittest.TestCase):
    def test_table2_exact(self):
        for r in table2():
            paper = PAPER_TABLE2[(r["rho"], round(r["delta"], 1), r["cutter"])]
            for a, b in zip((r["PI_1"], r["PI_static"], r["PI_dyn"]), paper):
                self.assertAlmostEqual(a, b, delta=6e-4)

    def test_dynamic_zero_for_any_sticky_nonpunisher(self):
        for rho in (0.1, 0.33, 0.7, 0.95):
            for cutter in ("returns", "stays low"):
                f = make_fork(rho, 0.0, cutter)
                self.assertAlmostEqual(pi_dynamic(f, rho), 0.0, places=10)

    def test_no_cut_scores_zero(self):
        p = np.full(16, 1.9)
        f = Fork(p, p, p, p)
        self.assertEqual(pi_one_period(f), 0.0)
        self.assertEqual(pi_static(f), 0.0)


class Stats(unittest.TestCase):
    def test_sign(self):
        self.assertAlmostEqual(sign_test_p([1] * 12)[2], 0.000488, places=5)
        self.assertAlmostEqual(sign_test_p([-1] * 9 + [1] * 3)[2], 0.1460, places=3)

    def test_rho_recovery(self):
        # deterministic partial adjustment toward BR + m, no rounding -> exact rho
        rho, m = 0.8, 0.3
        p = np.zeros((80, 2)); p[0] = [2.3, 1.7]
        for t in range(1, 80):
            for i in (0, 1):
                p[t, i] = rho * p[t - 1, i] + (1 - rho) * (m + best_response(p[t - 1, 1 - i]))
        self.assertAlmostEqual(pooled_rho([{"shown": p}], 1, 75), rho, places=6)


class Agents(unittest.TestCase):
    def test_parse(self):
        price, notes = parse_reply("<price>1.93</price><plans>hold</plans><insights>x</insights>")
        self.assertEqual(price, 1.93); self.assertEqual(notes.plans, "hold")

    def test_prompt_contains_history(self):
        s = render_prompt([Period(0, 1.9, 1.8, 0.3, 0.27)], Notes("p", "i"), punish_rule=True)
        self.assertIn("Competitor's price: 1.80", s); self.assertIn("1.50 for three rounds", s)

    def test_synthetic_lag(self):
        a = SyntheticAgent(rho=0.0, m=0.25, delta=0.4, K=3, lag=1)
        h = [Period(0, 2.0, 2.0, 0, 0), Period(1, 2.0, 1.8, 0, 0)]
        d1 = a.act(h, Notes())                    # sees cut, waits
        h.append(Period(2, d1.price, 2.0, 0, 0))
        d2 = a.act(h, d1.notes)                   # now punishes
        self.assertGreater(d1.price - (0.25 + best_response(1.8)), -0.006)
        self.assertLess(d2.price, 0.25 + best_response(2.0) - 0.39)


if __name__ == "__main__":
    unittest.main(verbosity=2)
