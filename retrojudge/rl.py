"""Reward-driven student: Double DQN on the harness's object features, in plain NumPy (no PyTorch).

Why this beats copying Laya: imitation can only approach its teacher; Q-learning optimises the game score itself
(Bellman target r + γ·max Q(s')). To learn fast it starts from Laya's game (demonstrations, DQfD-style margin
loss) and then improves with its own play, replayed from a buffer ("dreaming" over stored experience).
"""
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from .harness import MOVES, features

N_ACT = len(MOVES)


class QNet:
    """19 -> 64 -> 64 -> 3 MLP with Adam. Forward pass is ~0.05 ms."""

    def __init__(self, n_in, hidden=64, lr=5e-4, seed=0):
        r = np.random.default_rng(seed)
        he = lambda a, b: r.normal(0, np.sqrt(2 / a), (a, b))
        self.p = [he(n_in, hidden), np.zeros(hidden), he(hidden, hidden), np.zeros(hidden), he(hidden, N_ACT) * 0.1, np.zeros(N_ACT)]
        self.m = [np.zeros_like(x) for x in self.p]; self.v = [np.zeros_like(x) for x in self.p]
        self.lr, self.t = lr, 0

    def q(self, X):
        W1, b1, W2, b2, W3, b3 = self.p
        h1 = np.maximum(X @ W1 + b1, 0); h2 = np.maximum(h1 @ W2 + b2, 0)
        return h2 @ W3 + b3, (X, h1, h2)

    def step(self, cache, dq):
        X, h1, h2 = cache; W1, b1, W2, b2, W3, b3 = self.p
        g3 = h2.T @ dq; dh2 = (dq @ W3.T) * (h2 > 0)
        g2 = h1.T @ dh2; dh1 = (dh2 @ W2.T) * (h1 > 0)
        g1 = X.T @ dh1
        grads = [g1, dh1.sum(0), g2, dh2.sum(0), g3, dq.sum(0)]
        self.t += 1
        for i, g in enumerate(grads):
            g = np.clip(g, -5, 5)
            self.m[i] = 0.9 * self.m[i] + 0.1 * g; self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t); vh = self.v[i] / (1 - 0.999 ** self.t)
            self.p[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

    def copy(self):
        c = QNet.__new__(QNet); c.p = [x.copy() for x in self.p]
        return c


class DQNAgent:
    def __init__(self, n_in, gamma=0.99, nstep=3, batch=64, buffer=60_000, margin=0.8, demo_weight=1.0, seed=0):
        self.q, self.gamma, self.n, self.batch, self.margin, self.lam = QNet(n_in, seed=seed), gamma, nstep, batch, margin, demo_weight
        self.target = self.q.copy()
        self.buf, self.demo = deque(maxlen=buffer), []
        self.rng, self.updates, self.demo_frac = np.random.default_rng(seed), 0, 0.25

    # ---- transitions: (x, a, R_n, x_n, done_n, n_used) --------------------------------------------
    def nstep(self, traj):
        out = []
        for t in range(len(traj)):
            R, k, done = 0.0, 0, False
            for k in range(self.n):
                if t + k >= len(traj):
                    break
                x, a, r, x2, d = traj[t + k]
                R += (self.gamma ** k) * r
                if d:
                    done = True; break
            last = traj[min(t + k, len(traj) - 1)]
            out.append((traj[t][0], traj[t][1], R, last[3], done, k + 1))
        return out

    def train(self, n_updates=1):
        for _ in range(n_updates):
            use_demo = self.demo and (not self.buf or self.rng.random() < self.demo_frac)
            pool = self.demo if use_demo else self.buf
            idx = self.rng.integers(len(pool), size=self.batch)
            X = np.stack([pool[i][0] for i in idx]); A = np.array([pool[i][1] for i in idx])
            R = np.array([pool[i][2] for i in idx]); X2 = np.stack([pool[i][3] for i in idx])
            D = np.array([pool[i][4] for i in idx], float); K = np.array([pool[i][5] for i in idx])
            q, cache = self.q.q(X)
            a2 = self.q.q(X2)[0].argmax(1)                                  # Double DQN: online picks,
            y = R + (self.gamma ** K) * (1 - D) * self.target.q(X2)[0][np.arange(len(A)), a2]  # target evaluates
            td = q[np.arange(len(A)), A] - y
            dq = np.zeros_like(q); dq[np.arange(len(A)), A] = np.clip(td, -1, 1) / len(A)   # Huber gradient
            if use_demo:  # large-margin loss: the demonstrator's action should score at least `margin` higher
                aug = q + self.margin; aug[np.arange(len(A)), A] -= self.margin
                j = aug.argmax(1); m = j != A
                dq[np.arange(len(A))[m], j[m]] += self.lam / len(A)
                dq[np.arange(len(A))[m], A[m]] -= self.lam / len(A)
            self.q.step(cache, dq)
            self.updates += 1
            if self.updates % 1000 == 0:
                self.target = self.q.copy()

    def act(self, x, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(N_ACT))
        return int(self.q.q(x[None])[0].argmax())

    def save(self, path):
        np.savez(path, *self.q.p)

    def load(self, path):
        d = np.load(path); self.q.p = [d[f"arr_{i}"] for i in range(len(d.files))]; self.target = self.q.copy()


def rollout(h, choose, max_decisions=3000, start_noops=0, on_decision=None, stop=None):
    """Play one game; returns (trajectory, score, decisions). Reward = score gained, -25 for a lost life
    (scaled /25 so one kill ≈ 0.2–1.2 and a death = -1)."""
    obs = h._reset()
    for _ in range(start_noops):
        obs, _, _ = h._step([])
    s = h.state(obs); x = features(s); traj, score, lives = [], 0.0, h.lives()
    for d in range(max_decisions):
        a = choose(s, x); btn = {0: "LEFT", 2: "RIGHT"}.get(a)
        r, done = 0.0, False
        for f in range(h.k):
            obs, rr, done = h._step(([btn] if btn else []) + (["FIRE"] if f < h.k // 2 else []))
            r += rr
            if done:
                break
        score += r
        s2 = h.state(obs); x2 = features(s2); lost = h.lives() < lives; lives = h.lives()
        traj.append((x, a, (r - 25.0 * lost) / 25.0, x2, done))
        if on_decision:
            on_decision(obs, dict(decision=d + 1, move=MOVES[a], score=score, lives=lives, ms=0.0))
        s, x = s2, x2
        if done or (stop and stop()):
            break
    return traj, score, d + 1


def train(game="space_invaders", episodes=60, demo_log=None, eval_every=10, out="results/rl", seed=0, log=print,
          sticky=0.25, eval_games=8):
    from .harness import AtariHarness, make_harness
    h = AtariHarness(game, sticky=sticky, seed=seed) if not game.endswith("-v0") else make_harness(game)
    n_in = len(features(h.state(h._reset())))
    ag = DQNAgent(n_in, seed=seed)
    Path(out).mkdir(parents=True, exist_ok=True)
    if demo_log and Path(demo_log).exists():  # replay the teacher's game to recover its rewards
        L = [json.loads(l) for l in open(demo_log, encoding="utf-8")]
        moves = iter([MOVES.index(x["move"]) for x in L])
        # replay on a NON-sticky copy: the teacher's moves only match its states in the deterministic game
        hd = AtariHarness(game, seed=seed) if not game.endswith("-v0") else h
        traj, sc, _ = rollout(hd, lambda s, x: next(moves, 1), len(L))
        ag.demo = ag.nstep(traj)
        ag.train(3000)  # pre-train: start at the teacher's level before playing
        log(f"demo: replayed teacher game ({sc:.0f} pts, {len(traj)} moves) and pre-trained 3000 updates")
    curve, steps, t0, best = [], 0, time.time(), -1.0
    for ep in range(1, episodes + 1):
        eps = max(0.02, 0.3 * (1 - ep / (0.6 * episodes)))
        ag.demo_frac = 0.25 * max(0.0, 1 - ep / (0.5 * episodes))  # teacher's pull fades; the score takes over
        ag.lam = ag.demo_frac * 4
        traj, sc, n = rollout(h, lambda s, x: ag.act(x, eps), start_noops=int(ag.rng.integers(0, 30)))
        ag.buf.extend(ag.nstep(traj)); steps += n
        ag.train(n)  # one update per move played
        if ep % eval_every == 0 or ep == episodes:
            ev = [rollout(h, lambda s, x: ag.act(x))[1] for _ in range(eval_games)]
            m = float(np.mean(ev))
            curve.append(dict(episode=ep, train_score=sc, eval_mean=m, eval_sd=float(np.std(ev)), steps=steps,
                              minutes=round((time.time() - t0) / 60, 1)))
            log(json.dumps(curve[-1]))
            if m > best:  # keep the best version, not the last
                best = m; ag.save(Path(out) / f"{game}_dqn.npz")
    json.dump(curve, open(Path(out) / f"{game}_curve{'_demo' if demo_log else ''}.json", "w"), indent=1)
    return ag, curve


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("retrojudge.rl")
    ap.add_argument("--game", default="space_invaders"); ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--demo", help="teacher move log (e.g. results/retro/space_invaders_laya_ep0_log.jsonl)")
    ap.add_argument("--eval-every", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a.game, a.episodes, a.demo, a.eval_every, seed=a.seed, log=lambda m: print(m, flush=True))
