"""Retro-Judge viewer: watch agents play live, run checks.  python -m retrojudge ui

Local only (127.0.0.1). The browser can't run the emulator, Laya or Python, so this server does,
and streams game frames as PNGs. Only whitelisted jobs can run; nothing arbitrary.
"""
import json
import os
import struct
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

def _jsonable(o):
    """NumPy scalars/arrays (bool_, float64, ...) -> plain Python for JSON replies."""
    return o.item() if hasattr(o, "item") and getattr(o, "ndim", 0) == 0 else (o.tolist() if hasattr(o, "tolist") else str(o))


ROOT = Path(__file__).resolve().parent.parent  # project folder (tests/ and results/ live here)
PY = sys.executable
JOBS_SPEC = {
    "tests": dict(title="Run all tests", cmd=[PY, "-m", "unittest", "discover", "-s", "tests", "-v"]),
    "rl": dict(title="Train the RL player (DQN)", cmd=[PY, "-m", "retrojudge", "train", "--episodes", "300",
                                                     "--demo", "results/retro/space_invaders_laya_ep0_log.jsonl"]),
}
AGENTS = ["tracker", "random", "laya", "student_laya", "dqn"]
GAMES = {"space_invaders": "Space Invaders (Atari, works on Windows)", "Airstriker-Genesis-v0": "Airstriker (Genesis, Linux/macOS)"}


def laya_log(game):
    new = ROOT / "results" / "retro" / f"{game}_laya_ep0_log.jsonl"
    old = ROOT / "results" / "retro" / "laya_ep0_log.jsonl"  # Airstriker log from earlier versions
    return new if new.exists() or game != "Airstriker-Genesis-v0" else old
jobs, lock = {}, threading.Lock()
game = dict(running=False, frame=None, info={}, result=None, error=None, stop=False, trace=[])


def png(a):
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    ch = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw, 3)) + ch(b"IEND", b"")


def start_job(name, params):
    spec = JOBS_SPEC[name]
    cmd = list(spec["cmd"])
    jid = uuid.uuid4().hex[:8]
    j = dict(id=jid, name=name, title=spec["title"], status="running", lines=[], started=time.time(), ended=None, code=None)
    with lock:
        jobs[jid] = j

    def run():
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        try:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", env=env)
            j["pid"] = p.pid
            for line in p.stdout:
                j["lines"].append(line.rstrip("\n"))
                del j["lines"][:-2000]
            j["code"] = p.wait()
            j["status"] = "passed" if j["code"] == 0 else "failed"
        except Exception as e:
            j["lines"].append(f"Could not start: {e}"); j["status"] = "failed"
        j["ended"] = time.time()
    threading.Thread(target=run, daemon=True).start()
    return jid


MIN_LABELS = 150


def save_labels(game_id, rows):
    path = laya_log(game_id)
    if game_id == "Airstriker-Genesis-v0" and path.name == "laya_ep0_log.jsonl":  # migrate the old file name
        path = ROOT / "results" / "retro" / f"{game_id}_laya_ep0_log.jsonl"
        old = ROOT / "results" / "retro" / "laya_ep0_log.jsonl"
        if old.exists() and not path.exists():
            path.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in rows)


def start_game(agent, realtime=True, max_decisions=1500, game_id="space_invaders"):
    if game["running"]:
        raise ValueError("A game is already running. Stop it first.")
    if agent not in AGENTS:
        raise ValueError("unknown agent")
    if game_id not in GAMES:
        raise ValueError("unknown game")
    game.update(running=True, frame=None, info={}, result=None, error=None, stop=False, trace=[], agent=agent, game=game_id, note="")

    def run():
        try:
            import numpy as np
            from .harness import ReflexPolicy, laya_policy, make_harness, random_policy, tracker_policy
            h = make_harness(game_id)
            if agent in ("laya", "student_laya"):
                import laya
                teacher = laya_policy(laya.load("convaiinnovations/laya"))
            if agent == "tracker":
                pol = tracker_policy
            elif agent == "random":
                pol = random_policy(np.random.default_rng())
            elif agent == "laya":
                pol = teacher
            elif agent == "dqn":
                from .harness import MOVES, features
                from .rl import DQNAgent
                w = ROOT / "results" / "rl" / f"{game_id}_dqn.npz"
                if not w.exists():
                    raise ValueError("No trained RL player yet. Run: python -m retrojudge train")
                ag = DQNAgent(len(features(h.state(h._reset())))); ag.load(w)
                pol = lambda s: MOVES[ag.act(features(s))]
            else:
                log = laya_log(game_id)
                L = [json.loads(l) for l in open(log, encoding="utf-8")] if log.exists() else []
                if len(L) < MIN_LABELS:
                    raise ValueError(f"The student needs at least {MIN_LABELS} Laya moves to learn from; it has {len(L)}. "
                                     "Play Laya longer first (don't stop it early).")
                pol = ReflexPolicy(teacher).fit([x["state"] for x in L], [x["move"] for x in L])
                game["note"] = (f"Student trained on {len(L)} Laya moves. " + (
                    f"It acts alone when at least {pol.tau:.0%} sure." if pol.tau is not None else
                    "It isn't reliable yet, so it will ask Laya every time; each answer is saved for next run."))
            last = [time.perf_counter()]

            def on_decision(obs, info):
                if hasattr(pol, "calls"):
                    info["teacher_calls"] = pol.calls
                game["frame"], game["info"] = obs.copy(), info
                game["trace"].append(info["move"]); del game["trace"][:-120]
                if realtime:  # k frames per decision at 60 fps
                    wait = h.k / 60 - (time.perf_counter() - last[0])
                    if wait > 0:
                        time.sleep(wait)
                last[0] = time.perf_counter()
            moves = [] if agent == "laya" else None
            r = h.episode(pol, max_decisions, log=moves, on_decision=on_decision, stop=lambda: game["stop"])
            # Close the loop: every Laya answer (from a Laya game or from a student asking) is APPENDED,
            # so the next student run trains on more data and should need Laya less.
            new = moves if agent == "laya" else ([dict(state=st, move=m) for st, m in pol.new] if agent == "student_laya" else [])
            if new:
                save_labels(game_id, new)
                r["labels_added"] = len(new)
            if hasattr(pol, "calls"):
                r["teacher_call_rate"] = round(pol.calls / max(pol.total, 1), 3)
            r["agent"] = agent
            game["result"] = r
            out = ROOT / "results" / "retro"; out.mkdir(parents=True, exist_ok=True)
            with open(out / "ui_games.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
        except ModuleNotFoundError as e:
            pkg = {"ale_py": "ale-py", "stable_retro": "stable-retro", "laya": "laya", "torch": "torch"}.get(e.name, e.name)
            game["error"] = f"Missing package '{e.name}'. Install it with: pip install {pkg}"
        except Exception as e:
            game["error"] = f"{type(e).__name__}: {e}"
        finally:
            game["running"] = False
    threading.Thread(target=run, daemon=True).start()


def results():
    out = []
    for p in sorted((ROOT / "results").rglob("*.json")):
        try:
            out.append(dict(name=str(p.relative_to(ROOT / "results")), data=json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            pass
    for p in sorted((ROOT / "results" / "retro").glob("*.jsonl")) if (ROOT / "results" / "retro").exists() else []:
        if "_log" not in p.name:
            rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            out.append(dict(name=str(p.relative_to(ROOT / "results")), data=rows[-10:]))
    return out


def label_count(game_id):
    p = laya_log(game_id)
    return sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body, default=_jsonable).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def local(self):  # blocks DNS-rebinding: only our own host names are served
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def do_GET(self):
        if not self.local():
            return self.send(403, dict(error="local access only"))
        path = self.path.split("?")[0]
        if path == "/":
            return self.send(200, (Path(__file__).parent / "ui.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            with lock:
                js = [{k: v for k, v in j.items() if k != "lines"} | dict(tail=j["lines"][-1] if j["lines"] else "") for j in jobs.values()]
            g = {k: game.get(k) for k in ("running", "info", "result", "error", "trace", "agent", "game", "note")}
            return self.send(200, dict(jobs=js, game=g,
                                       laya_logs={k: label_count(k) for k in GAMES}, min_labels=MIN_LABELS))
        if path.startswith("/api/job/"):
            j = jobs.get(path.rsplit("/", 1)[-1])
            return self.send(200, j) if j else self.send(404, dict(error="no such job"))
        if path == "/api/frame":
            f = game.get("frame")
            return self.send(200, png(f), "image/png") if f is not None else self.send(204, b"", "image/png")
        if path == "/api/results":
            return self.send(200, results())
        self.send(404, dict(error="not found"))

    def do_POST(self):
        if not self.local():
            return self.send(403, dict(error="local access only"))
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            return self.send(415, dict(error="JSON only"))
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
            if self.path == "/api/run":
                if req.get("job") not in JOBS_SPEC:
                    raise ValueError("unknown job")
                return self.send(200, dict(id=start_job(req["job"], req)))
            if self.path == "/api/game/start":
                start_game(req.get("agent", "tracker"), bool(req.get("realtime", True)), int(req.get("max_decisions", 1500)),
                           req.get("game", "space_invaders"))
                return self.send(200, dict(ok=True))
            if self.path == "/api/game/stop":
                game["stop"] = True; return self.send(200, dict(ok=True))
            self.send(404, dict(error="not found"))
        except (ValueError, KeyError, TypeError) as e:
            self.send(400, dict(error=str(e)))


def main(port=8790, open_browser=True):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Retro-Judge viewer running at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
