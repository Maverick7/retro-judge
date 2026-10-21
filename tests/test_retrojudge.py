import unittest

import numpy as np

from retrojudge.calibrate import conformal_qhat, cp_upper, fit_temperature, risk_threshold, softmax


class TestCalibration(unittest.TestCase):
    def test_temperature_recovers_overconfidence(self):
        rng = np.random.default_rng(0); y = rng.integers(0, 3, 4000)
        z = rng.normal(0, 1, (4000, 3)); z[np.arange(4000), y] += 1.0
        self.assertGreater(fit_temperature(z * 4, y), 2.0)  # 4x-sharpened logits need T>2

    def test_cp_bound_and_risk_threshold_controls_error(self):
        self.assertLess(cp_upper(0, 200, 0.05), 0.02); self.assertEqual(cp_upper(5, 5, .05), 1.0)
        rng = np.random.default_rng(1); conf = rng.uniform(0, 1, 20000); correct = rng.uniform(0, 1, 20000) < conf
        tau = risk_threshold(conf[:5000], correct[:5000], alpha=0.1, delta=0.05)
        m = conf[5000:] >= tau
        self.assertLessEqual(1 - correct[5000:][m].mean(), 0.1)

    def test_risk_threshold_refuses_when_impossible(self):
        self.assertIsNone(risk_threshold(np.ones(500), np.zeros(500, bool), 0.05, 0.05))

    def test_conformal_coverage(self):
        rng = np.random.default_rng(2); y = rng.integers(0, 5, 6000)
        z = rng.normal(0, 1, (6000, 5)); z[np.arange(6000), y] += 1.5; p = softmax(z)
        q = conformal_qhat(p[:3000], y[:3000], 0.1)
        cov = np.mean(p[3000:][np.arange(3000), y[3000:]] >= 1 - q)
        self.assertGreater(cov, 0.88)


class TestStudent(unittest.TestCase):
    def states(self, n=300, seed=0):
        rng = np.random.default_rng(seed); S, M = [], []
        for _ in range(n):
            dx, up = int(rng.integers(-120, 120)), int(rng.integers(5, 180))
            S.append(dict(me_x=int(rng.integers(20, 300)), lives=3, score=0, objects=[dict(dx=dx, above=up, size=40)]))
            M.append("move_right" if (up < 50 and -16 < dx <= 0) else "stay")  # one-sided teacher, like Laya
        return S, M

    def test_mirror_teaches_missing_direction_and_escalation_counts(self):
        from retrojudge.harness import ReflexPolicy
        S, M = self.states()
        calls = []
        p = ReflexPolicy(lambda s: calls.append(s) or "stay", risk=0.2).fit(S, M)
        threat_right = dict(me_x=150, lives=3, score=0, objects=[dict(dx=8, above=20, size=40)])
        p.tau = 0.0  # student only
        self.assertEqual(p(threat_right), "move_left")  # teacher never showed a left dodge
        p.tau = 1.1  # force escalation
        p(threat_right); self.assertEqual((p.calls, len(p.new)), (1, 1))


class TestAtariHarness(unittest.TestCase):
    def test_space_invaders_probe_and_play(self):
        try:
            import ale_py  # noqa: F401
        except ImportError:
            self.skipTest("ale-py not installed")
        from retrojudge.harness import AtariHarness, tracker_policy
        h = AtariHarness("space_invaders")
        self.assertGreater(h.x_corr, 0.9)  # found the byte that tracks the player's sprite
        s = h.state(h._reset())
        self.assertEqual((s["width"], s["lives"]), (160, 3))
        r = h.episode(tracker_policy, 300)
        self.assertGreater(r["score"], 0)


class TestDQN(unittest.TestCase):
    def test_qnet_learns_a_bandit_and_margin_prefers_demo_action(self):
        from retrojudge.rl import DQNAgent
        rng = np.random.default_rng(0); ag = DQNAgent(4, batch=32, seed=0)
        X = rng.normal(size=(400, 4)).astype(float)
        # action 2 pays +1, others 0; one-step episodes
        ag.buf.extend((x, int(a), 1.0 if a == 2 else 0.0, x, True, 1) for x, a in zip(X, rng.integers(0, 3, 400)))
        ag.train(1500)
        self.assertTrue((ag.q.q(X)[0].argmax(1) == 2).mean() > 0.9)
        ag2 = DQNAgent(4, batch=32, seed=1)
        ag2.demo = [(x, 0, 0.0, x, True, 1) for x in X]  # demonstrator always picks 0, no reward signal
        ag2.train(800)
        self.assertTrue((ag2.q.q(X)[0].argmax(1) == 0).mean() > 0.9)


class TestCli(unittest.TestCase):
    def test_help_and_unknown_agent(self):
        from retrojudge.cli import main
        with self.assertRaises(SystemExit):
            main(["judge", "--agent", "nope"])


if __name__ == "__main__":
    unittest.main()
