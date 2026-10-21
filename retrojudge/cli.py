"""python -m retrojudge {play,judge,train,ui}"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def make_policy(agent, h, game):
    from .harness import MOVES, features, laya_policy, random_policy, tracker_policy
    if agent == "random":
        return random_policy(np.random.default_rng(0))
    if agent == "stay":
        return lambda s: "stay"
    if agent == "tracker":
        return tracker_policy
    if agent == "laya":
        import laya
        return laya_policy(laya.load("convaiinnovations/laya"))
    if agent == "dqn":
        from .rl import DQNAgent
        w = ROOT / "results" / "rl" / f"{game}_dqn.npz"
        if not w.exists():
            raise FileNotFoundError(f"{w} missing. Train first: python -m retrojudge train")
        ag = DQNAgent(len(features(h.state(h._reset())))); ag.load(w)
        return lambda s: MOVES[ag.act(features(s))]
    raise ValueError(agent)


def judge(agent, game="space_invaders", games=50, max_decisions=3000, sticky=0.25, out="results/judge"):
    """Fresh seeds, sticky actions (ALE standard 0.25): the score you report, not the best run you saw."""
    from .harness import AtariHarness
    scores = []
    for i in range(games):
        h = AtariHarness(game, sticky=sticky, seed=1000 + i)
        scores.append(h.episode(make_policy(agent, h, game), max_decisions)["score"])
        print(f"game {i + 1}/{games}: {scores[-1]:.0f}", flush=True)
    r = dict(agent=agent, game=game, games=games, sticky=sticky, mean=float(np.mean(scores)),
             sd=float(np.std(scores)), min=float(np.min(scores)), max=float(np.max(scores)))
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / f"{game}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(r) + "\n")
    return r


def main(argv=None):
    ap = argparse.ArgumentParser("retrojudge", description="Can typed decision models play retro games?")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play", help="play a few episodes and log every move")
    p.add_argument("rest", nargs=argparse.REMAINDER, help="passed to the harness: --game --agent --episodes ...")
    j = sub.add_parser("judge", help="mean ± sd over many fresh sticky-action games")
    j.add_argument("--agent", choices=["random", "stay", "tracker", "laya", "dqn"], default="tracker")
    j.add_argument("--game", default="space_invaders"); j.add_argument("--games", type=int, default=50)
    t = sub.add_parser("train", help="train the NumPy Double DQN (optionally from a teacher's move log)")
    t.add_argument("--game", default="space_invaders"); t.add_argument("--episodes", type=int, default=300)
    t.add_argument("--demo", help="e.g. results/retro/space_invaders_laya_ep0_log.jsonl"); t.add_argument("--seed", type=int, default=0)
    u = sub.add_parser("ui", help="watch agents play live in your browser")
    u.add_argument("--port", type=int, default=8790); u.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "play":
            from .harness import main as play
            play(a.rest)
        elif a.cmd == "judge":
            print(json.dumps(judge(a.agent, a.game, a.games), indent=2))
        elif a.cmd == "train":
            from .rl import train
            train(a.game, a.episodes, a.demo, 20, seed=a.seed, log=lambda m: print(m, flush=True))
        elif a.cmd == "ui":
            from .ui import main as ui
            ui(a.port, not a.no_browser)
    except ModuleNotFoundError as e:
        pkg = {"ale_py": "ale-py", "stable_retro": "stable-retro"}.get(e.name, e.name)
        print(f"error: missing package '{e.name}'. pip install {pkg}", file=sys.stderr); return 2
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr); return 2
    return 0
