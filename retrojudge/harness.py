"""Laya plays retro games through a generic harness.

Nothing about the game's rules is given to the agent. The harness only:
  1. probes RAM to find which bytes move when the agent presses LEFT/RIGHT (its own position),
  2. turns pixels into an object list (dark-background games: sprites = bright blobs),
  3. describes that as JSON-ish text and asks a typed choice question.
Backends: Atari 2600 via ale-py (Windows/Linux/macOS, ROMs bundled) and Genesis via stable-retro (Linux/macOS).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

MOVES = ["move_left", "stay", "move_right"]


class Harness:
    """Shared logic. Subclasses set: width, height, play=(top, bottom), band=(y0, y1), ram_bytes, k, and
    implement _reset() -> obs, _step(names) -> (obs, reward, done), _ram(), lives()."""
    probe_steps, warmup = 40, 0  # warmup: frames a game ignores input after reset

    def setup(self):
        self.x_addr, self.offset = self.probe()

    # ---- 1. discover "where am I" from RAM, no game knowledge --------------------------------
    def probe(self):
        self._reset()
        for _ in range(self.warmup):
            self._step([])
        out = []
        for btn in ("RIGHT", "LEFT"):  # one run: right then left, so a player starting at an edge still moves
            for _ in range(self.probe_steps):
                self._step([btn]); out.append(self._vals(self._ram()))
        d = np.diff(np.array(out), axis=0); n = self.probe_steps
        dr, dl = d[5:n - 1], d[n + 5:]
        cand = [i for i in range(dr.shape[1]) if (dr[:, i] >= 0).all() and dr[:, i].sum() > 5
                and (dl[:, i] <= 0).all() and dl[:, i].sum() < -5]
        if not cand:
            raise RuntimeError("could not find a RAM byte that follows LEFT/RIGHT")
        # several bytes can follow the joystick; keep the one that tracks the player's sprite on screen
        self._reset()
        for _ in range(self.warmup):
            self._step([])
        rows, blob = [], []
        for t in range(60):
            obs, _, _ = self._step(["RIGHT" if (t // 15) % 2 == 0 else "LEFT"])
            me = self._me(self.objects(obs))
            if me:
                blob.append(me[0]); rows.append(self._vals(self._ram()))
        best, best_c = cand[0], -1.0
        for a in cand:
            v = np.array([r[a] for r in rows])
            c = abs(np.corrcoef(v, blob)[0, 1]) if len(v) > 5 and v.std() > 0 else 0.0
            if c > best_c:
                best, best_c = a, c
        self.x_addr, self.x_corr = best, best_c
        v = np.array([r[best] for r in rows])
        return best, float(np.median(np.array(blob) - v)) if len(v) else 0.0

    def _vals(self, r):
        """Every RAM address read as a number (16-bit little-endian words for Genesis)."""
        r = r.astype(np.int64)
        return r + 256 * np.append(r[1:], 0) if self.ram_bytes == 2 else r

    def ram_x(self):
        r = self._ram(); a = self.x_addr
        return int(r[a]) + (256 * int(r[a + 1]) if self.ram_bytes == 2 else 0)

    # ---- 2. pixels -> objects -------------------------------------------------------------------
    def objects(self, obs):
        top, bottom = self.play
        m = obs[top:bottom].max(2) > 60
        lab, _ = ndimage.label(m)
        out = []
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            size = int((lab[sl] == i).sum())
            if size >= 6:  # drop tiny specks
                out.append(((sl[1].start + sl[1].stop) / 2, (sl[0].start + sl[0].stop) / 2 + top, size))
        return out

    def _in_band(self, o):
        return self.band[0] <= o[1] <= self.band[1]

    def _me(self, objs):
        b = [o for o in objs if self._in_band(o)]
        return max(b, key=lambda o: o[2]) if b else None

    def state(self, obs):
        me_x = self.ram_x() + self.offset
        objs = self.objects(obs)
        me = min((o for o in objs if self._in_band(o)), key=lambda o: abs(o[0] - me_x), default=None)
        me_y = me[1] if me else float(self.band[1])
        others = [o for o in objs if o is not me and o[1] < me_y - 4]
        others.sort(key=lambda o: (me_y - o[1]) + abs(o[0] - me_x))  # nearest first
        return dict(me_x=round(me_x), width=self.width, lives=self.lives(),
                    objects=[dict(dx=round(o[0] - me_x), above=round(me_y - o[1]), size=o[2]) for o in others[:6]])

    # ---- 3. play --------------------------------------------------------------------------------
    def episode(self, policy, max_decisions=400, start_noops=0, log=None, on_decision=None, stop=None):
        obs = self._reset()
        for _ in range(start_noops):
            obs, _, _ = self._step([])
        score, lat, done, d = 0.0, [], False, 0
        for d in range(max_decisions):
            s = self.state(obs)
            t = time.perf_counter(); move = policy(s); lat.append((time.perf_counter() - t) * 1000)
            btn = {"move_left": "LEFT", "move_right": "RIGHT"}.get(move)
            for f in range(self.k):
                names = ([btn] if btn else []) + (["FIRE"] if f < self.k // 2 else [])
                obs, r, done = self._step(names)
                score += r
                if done:
                    break
            if log is not None:
                log.append(dict(state=s, move=move))
            if on_decision is not None:  # UI hook: live frame + decision
                on_decision(obs, dict(decision=d + 1, move=move, score=score, lives=self.lives(), ms=lat[-1]))
            if done or (stop is not None and stop()):
                break
        return dict(game=self.name, score=score, decisions=d + 1, frames=(d + 1) * self.k, lives_left=self.lives(),
                    decide_ms_p50=float(np.median(lat)))


class AtariHarness(Harness):
    """Atari 2600 via ale-py: native Windows wheels, ROMs bundled. Default: Space Invaders."""
    ACTIONS = {(): "NOOP", ("FIRE",): "FIRE", ("RIGHT",): "RIGHT", ("LEFT",): "LEFT",
               ("FIRE", "RIGHT"): "RIGHTFIRE", ("FIRE", "LEFT"): "LEFTFIRE"}

    def __init__(self, game="space_invaders", frames_per_decision=4, seed=0,
                 play=(20, 195), band=(176, 196), sticky=0.0):
        from ale_py import ALEInterface, LoggerMode, roms
        ALEInterface.setLoggerMode(LoggerMode.Error)
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", seed); self.ale.setFloat("repeat_action_probability", 0.0)
        self.ale.loadROM(roms.get_rom_path(game))
        self.sticky = sticky  # set after probing: 0.25 = standard ALE benchmark randomness
        self.A = {str(a).split(".")[-1]: a for a in self.ale.getMinimalActionSet()}
        self.name, self.k, self.play, self.band, self.ram_bytes = game, frames_per_decision, play, band, 1
        self.height, self.width = self.ale.getScreenRGB().shape[:2]
        self.warmup = 140
        self.setup()
        if sticky:  # probe deterministically, then play with sticky actions
            self.ale.setFloat("repeat_action_probability", sticky); self.ale.loadROM(roms.get_rom_path(game))

    def _reset(self):
        self.ale.reset_game(); self.prev = self.ale.getScreenRGB()
        return self.prev

    def _step(self, names):
        key = tuple(sorted(n for n in names if n in ("FIRE", "LEFT", "RIGHT")))
        a = self.A.get(self.ACTIONS.get(key, "NOOP"), self.A["NOOP"])
        r = self.ale.act(a)
        cur = self.ale.getScreenRGB()
        obs = np.maximum(cur, self.prev); self.prev = cur  # merge 2 frames: Atari sprites flicker
        return obs, float(r), self.ale.game_over()

    def _ram(self):
        return self.ale.getRAM()

    def lives(self):
        return int(self.ale.lives())


class RetroHarness(Harness):
    """Genesis via stable-retro (Linux/macOS/WSL2). Default: Airstriker, ROM bundled legally."""
    def __init__(self, game="Airstriker-Genesis-v0", fire="B", frames_per_decision=8, hud_rows=24, seed=0):
        import stable_retro as retro
        self.env = retro.make(game, render_mode=None)
        self.B, self.fire, self.k = self.env.buttons, fire, frames_per_decision
        self.name, self.ram_bytes, self.info = game, 2, {}
        self.height, self.width = 224, 320
        self.play, self.band = (hud_rows, 224), (168, 224)
        self.setup()

    def _act(self, names):
        a = np.zeros(len(self.B), np.int8)
        for n in names:
            a[self.B.index(self.fire if n == "FIRE" else n)] = 1
        return a

    def _reset(self):
        obs, self.info = self.env.reset()
        return obs

    def _step(self, names):
        obs, r, term, trunc, self.info = self.env.step(self._act(names))
        return obs, float(r), bool(term or trunc)

    def _ram(self):
        return self.env.get_ram()

    def lives(self):
        return self.info.get("lives")


def make_harness(game="space_invaders"):
    return RetroHarness(game) if game.endswith("-v0") else AtariHarness(game)


# ---- agents -----------------------------------------------------------------------------------
def random_policy(rng):
    return lambda s: MOVES[rng.integers(3)]


def tracker_policy(s):
    """Hand-written baseline: dodge anything close overhead, else line up under the nearest target."""
    near = [o for o in s["objects"] if o["above"] < 50 and abs(o["dx"]) < 16]
    if near:
        return "move_right" if near[0]["dx"] <= 0 else "move_left"
    if s["objects"]:
        dx = s["objects"][0]["dx"]
        return "stay" if abs(dx) < 6 else ("move_right" if dx > 0 else "move_left")
    return "stay"


QUESTION = {"move": {"type": "choice",
                     "instructions": ("You pilot a ship at the bottom of a vertical shooter; your shots fly straight up. "
                                      "Objects list other things on screen: dx is horizontal offset (negative = to your left), "
                                      "above is pixels above you. Avoid being hit from above, and line up under enemies to shoot them."),
                     "criteria": {"move_left": "move left", "stay": "stay in place", "move_right": "move right"}}}


def describe(s):
    return json.dumps({"lives": s["lives"], "objects": s["objects"]})


def laya_policy(agent, question=QUESTION):
    return lambda s: agent.predict(describe(s), question)["answers"]["move"]["choice"]


def main(argv=None):
    ap = argparse.ArgumentParser("retrojudge.harness")
    ap.add_argument("--game", default="space_invaders", help="Atari ROM name (ale-py) or a stable-retro id ending in -v0")
    ap.add_argument("--agent", choices=["random", "tracker", "laya"], default="tracker")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-decisions", type=int, default=400)
    ap.add_argument("--laya-model", default="convaiinnovations/laya")
    ap.add_argument("--out", default="results/retro")
    a = ap.parse_args(argv)
    h = make_harness(a.game)
    print(f"probe: player x at RAM[{h.x_addr}] ({8 * h.ram_bytes}-bit, corr {h.x_corr:.2f}), screen offset {h.offset:+.0f}", flush=True)
    if a.agent == "laya":
        import laya
        pol = laya_policy(laya.load(a.laya_model))
    else:
        pol = tracker_policy if a.agent == "tracker" else random_policy(np.random.default_rng(0))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for ep in range(a.episodes):
        log = []
        r = h.episode(pol, a.max_decisions, start_noops=ep * 7, log=log)
        r.update(agent=a.agent, episode=ep)
        print(json.dumps(r), flush=True)
        with open(Path(a.out) / f"{a.game}_{a.agent}.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        with open(Path(a.out) / f"{a.game}_{a.agent}_ep{ep}_log.jsonl", "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in log)



# ---- distillation: a ~0.05 ms student that asks its teacher only when unsure ---------------------
def features(s, k=3):
    v = []
    objs = s["objects"] + [dict(dx=0, above=300, size=0)] * k
    for o in objs[:k]:
        dx, up = o["dx"] / 160, o["above"] / 200
        v += [dx, up, abs(dx), dx * (up < 0.3), float(abs(o["dx"]) < 16), o["size"] / 300]
    return np.array(v + [s["me_x"] / s.get("width", 320)])


class ReflexPolicy:
    """Student = logistic regression on object features, temperature-calibrated on held-out teacher labels.
    Confident -> act in microseconds; unsure -> ask the teacher (Laya or any policy) and log the answer."""
    def __init__(self, teacher, risk=0.10, class_weight=None, mirror=True, model="trees"):
        self.teacher, self.risk, self.calls, self.total, self.new = teacher, risk, 0, 0, []
        self.class_weight, self.mirror, self.model = class_weight, mirror, model

    def _make(self):
        # Teachers decide with if-then rules; a linear model tops out ~89% agreement, trees reach ~99%.
        if self.model == "trees":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=5, class_weight=self.class_weight, random_state=0)
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=3.0, max_iter=3000, class_weight=self.class_weight)

    def fit(self, states, moves):
        from .calibrate import fit_temperature, risk_threshold, softmax
        if self.mirror:  # inject left/right symmetry the teacher may lack
            flip = {"move_right": "move_left", "move_left": "move_right", "stay": "stay"}
            states = list(states) + [dict(s, me_x=s.get("width", 320) - s["me_x"], objects=[dict(o, dx=-o["dx"]) for o in s["objects"]]) for s in states]
            moves = list(moves) + [flip[m] for m in moves]
        X = np.array([features(s) for s in states]); y = np.array([MOVES.index(m) for m in moves])
        idx = np.random.default_rng(0).permutation(len(y)); cut = int(0.7 * len(y))
        tr, ca = idx[:cut], idx[cut:]
        self.clf = self._make().fit(X[tr], y[tr])
        z = self._logits(X[ca]); self.T = fit_temperature(z, y[ca]); p = softmax(z / self.T)
        self.tau = risk_threshold(p.max(1), p.argmax(1) == y[ca], alpha=self.risk, delta=0.1)
        self.holdout_agreement = float((p.argmax(1) == y[ca]).mean())
        self.clf = self._make().fit(X, y)  # refit on everything; T and tau kept from the held-out split
        self.n_labels = len(y)
        return self

    def _logits(self, X):
        z = np.full((len(X), 3), -1e4)
        d = np.asarray(self.clf.decision_function(X))
        if d.ndim == 1:  # only two classes seen
            d = np.stack([-d / 2, d / 2], 1)
        z[:, self.clf.classes_] = d
        return z

    def __call__(self, s):
        from .calibrate import softmax
        self.total += 1
        p = softmax(self._logits(features(s)[None]) / self.T)[0]
        if self.tau is not None and p.max() >= self.tau:
            return MOVES[int(p.argmax())]
        self.calls += 1
        m = self.teacher(s); self.new.append((s, m))
        return m


if __name__ == "__main__":
    main()
