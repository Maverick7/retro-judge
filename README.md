# retro-judge

**Can typed decision models play retro games? An honest benchmark with Laya, DQN and imitation learning.**

> Status: research prototype. Tested on Windows 11 + Python 3.11 and Ubuntu. Issues welcome; no promises on support.

<!-- VIDEO: drop the demo clip here, e.g.
https://github.com/user-attachments/assets/<your-video-id>
-->
_Demo video coming soon._

## The three numbers

On **Space Invaders** (Atari 2600, sticky actions 0.25, fresh seeds — the standard ALE setup):

| Agent | What it is | Score |
|---|---|---|
| Scripted tracker | 20 lines: dodge bullets, track the nearest alien | **421.5 ± 39** (50 games) |
| Laya | ModernBERT decision model, asked "left / stay / right?" each move | 285 (one long game, 806 moves) |
| Double DQN | NumPy, no torch, learns from the score in ~10 min | **214 ± 43** (50 games; best training eval was 282) |

Baselines: random 30, always-stay 180.

**Headline:** a 20-line script beats both a language decision model and a from-scratch RL agent. That is the result, not a bug.

## Run it (3 commands)

```bash
pip install -e .                                   # ale-py ships the Space Invaders ROM
python -m retrojudge judge --agent tracker --games 50
python -m retrojudge ui                            # watch agents play live at 127.0.0.1:8790
```

More:

```bash
python -m retrojudge play --agent random --episodes 3      # play and log every move
python -m retrojudge train --episodes 300                  # train the DQN from scratch
python -m retrojudge train --demo results/retro/space_invaders_laya_ep0_log.jsonl   # DQN that starts from Laya's game
pip install -e ".[laya]"; python -m retrojudge judge --agent laya --games 5
python -m unittest discover -s tests -v
```

## How it works

```
emulator ──pixels+RAM──▶ harness ──typed state──▶ agent ──left/stay/right──▶ emulator
```

1. **Harness, no game rules given.** It presses LEFT/RIGHT and correlates RAM bytes with the moving sprite to find "where am I". Pixels become an object list (`dx`, `above`, `size`) by connected-component labelling.
2. **Agents** answer one typed question per move: `move ∈ {move_left, stay, move_right}`.
   - `tracker` — hand-written baseline.
   - `laya` — the state is described as text; Laya returns a calibrated choice.
   - `student_laya` — gradient-boosted trees distilled from Laya's moves, with mirror augmentation. It acts alone when it is confident enough (risk-controlled threshold) and asks Laya otherwise; every answer is appended to the log, DAgger style.
   - `dqn` — Double DQN with n-step returns and an optional demo margin loss that fades out as the score takes over.
3. **Judge.** `judge` plays many fresh sticky-action games and reports mean ± sd. One lucky run is not a result.

## What I learned

- **Winner's curse is real.** The DQN's best checkpoint scored 282 in training evals and 214 on 50 fresh games. Report the fresh number.
- **Distillation copies blind spots.** Laya never dodged left in 58 threat situations; the student only learned it after mirror augmentation.
- **A good student stops asking.** The tree student asked Laya on 0.4% of moves; the earlier linear student asked 100% of the time.
- **Latency matters.** Laya runs a transformer forward pass per move; tracker, student and DQN decide in well under 1 ms.

## Honest limits

- One main game (Space Invaders). Airstriker (Genesis) needs `stable-retro`, which has no Windows wheels; the logs are in `results/retro/`.
- Laya's number is a single long game, not 50.
- The object detector assumes a dark background with bright sprites.
- No ROMs are in this repo. Space Invaders comes with `ale-py`; bring your own legal ROMs for anything else.

## Repo map

```
retrojudge/harness.py   RAM probe, pixel→objects, Atari + Genesis backends, policies, distilled student
retrojudge/rl.py        NumPy Double DQN, rollout, training loop
retrojudge/calibrate.py temperature scaling, risk threshold (Clopper-Pearson), conformal
retrojudge/ui.py        local viewer (127.0.0.1 only) streaming frames as PNG
results/                move logs, learning curves, trained DQN weights
```

## History

Split out of a bigger experiment that started with a fruit-fly connectome (see [flydream](https://github.com/Maverick7/flydream)) and grew into [trustreflex](https://github.com/Maverick7/trustreflex). Real and shuffled fly wiring scored the same, so the connectome was dropped here.

## License

MIT © Dickson Xavier
