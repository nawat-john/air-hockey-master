# Air Hockey Bot (10 Hz, hybrid physics + RL)

**▶ Play in your browser:** https://nawat-john.github.io/air-hockey-master/
No install — plays against the bot on your half.

A 2D air-hockey simulator and a strong bot that **predicts the puck's
trajectory** (including wall bounces) and learns strategy with reinforcement
learning. It decides at **10 Hz** — like a human's reaction time — so it must
anticipate, not chase. See [`air_hockey_bot_plan.md`](air_hockey_bot_plan.md)
for the full design (Thai).

## Architecture

```
airhockey/
  config.py      TableConfig — geometry, dynamics, frequencies (all SI units)
  physics.py     AirHockeyPhysics — 200 Hz substep sim, no tunnelling, no teleport
  predictor.py   TrajectoryPredictor — mirror-unfolding intercept + aim (the "secret sauce")
  env.py         AirHockeyEnv — Gymnasium env, 10 Hz decisions, predicted features in obs
  opponents.py   ScriptedPredictor (baseline), PolicyOpponent (self-play), Random/Still
  render.py      Renderer — pygame view for human play and GIFs
scripts/
  play_human.py    play with the mouse vs the bot
  train_sac.py     SAC training with curriculum + self-play league
  evaluate.py      win rate and sub-metrics vs a baseline
  record_gif.py    capture a demo GIF
  export_policy.py dump the trained actor to docs/policy.js for the web game
tests/
  test_physics.py  collision / no-tunnelling / predictor-accuracy checks
```

The puck physics is solved analytically (it's just kinematics + reflections);
RL handles the hard part — *when* to attack vs defend, shot angle/power, and
coping with the 10 Hz decision latency. The trajectory predictor feeds the
policy as observation features and also serves as the scripted baseline opponent.

## Setup

Requires **Python 3.10+**. Create a virtual environment and install the deps:

```bash
python -m venv .venv

# activate it —
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\Activate.ps1       # Windows (PowerShell)

pip install -r requirements.txt
```

GPU training needs the **CUDA build of PyTorch**. If pip pulls the CPU build,
reinstall it (CUDA 12.4 shown — pick the build matching your driver):

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

Verify with `python -c "import torch; print(torch.cuda.is_available())"`.

## Quick start

With the virtual environment activated:

```bash
# Verify the physics first (always before training)
python -m pytest tests -q

# Play against the bot with your mouse
python scripts/play_human.py

# Train through the curriculum (each phase continues from the previous one)
python scripts/train_sac.py --phase defend   --timesteps 150000
python scripts/train_sac.py --phase attack   --timesteps 200000 --load runs/defend.zip
python scripts/train_sac.py --phase full     --timesteps 400000 --load runs/attack.zip
python scripts/train_sac.py --phase selfplay --timesteps 800000 --load runs/full.zip

# Evaluate and record a demo
python scripts/evaluate.py --model runs/sac_final.zip --episodes 200
python scripts/record_gif.py --model runs/sac_final.zip --out demo.gif

# Watch training curves
python -m tensorboard.main --logdir runs/tb
```

On headless machines (CI, servers) pygame has no display — prefix commands that
run the sim with `SDL_VIDEODRIVER=dummy`, e.g. `SDL_VIDEODRIVER=dummy python -m pytest tests -q`.

## Notes

- **Coordinates:** the agent is the LEFT (blue) mallet, defends `x=0`, always
  attacks `+x`. Self-play opponents are mirrored so a left-trained policy can
  drive the right mallet (`opponents.PolicyOpponent`).
- **Curriculum** (plan §5.3): `defend` (block serves) → `attack` (clear to the
  far side) → `full` game vs scripted → `selfplay` league with a checkpoint pool
  + scripted mix-in (PFSP-lite) so the agent never forgets the basics.
- **Reward** (plan §3.5): sparse goals (+10 / −10, own-goal −15) plus small
  shaping (possession, shooting toward +x, defensive positioning, anti-stall).
  Disable shaping with the env's `shaping=False` if you suspect reward hacking.

## Browser game (GitHub Pages)

`docs/` is a self-contained static web game (HTML + CSS + vanilla JS) — the
physics, trajectory predictor, and scripted bot are ported to JavaScript so it
runs entirely client-side with **no backend**. GitHub Pages can't run PyTorch, so
the trained **RL policy is shipped as weights, not a runtime**: `export_policy.py`
dumps the SAC actor (16→[256,256]→2) to `docs/policy.js`, and the **`RL (trained)`**
difficulty runs that net via a hand-coded MLP forward pass in `game.js`. Re-run
the export after retraining to refresh the web policy:

```bash
python scripts/export_policy.py --model runs/sac_final.zip
```

Test it locally before pushing:

```bash
python -m http.server -d docs 8000   # then open http://localhost:8000
```

To publish, push to GitHub then enable Pages: **Settings → Pages → Build and
deployment → Deploy from a branch**, branch `main`, folder **`/docs`**. The game
goes live at `https://<owner>.github.io/<repo>/` after ~1 min.

Difficulty `easy/normal/hard` scales the scripted bot's speed, aggression, and
reaction delay; `RL (trained)` instead runs the trained policy client-side. The
JS physics mirrors `airhockey/physics.py` 1:1, so behaviour matches the Python
sim, and the ported actor matches `model.predict` to ~1e-4.

## Shipping the trained model

Model weights are gitignored (`runs/`, `*.zip`, `*.gif`) to keep binaries out of
git history. Trained models are tiny (~3 MB), so they're published as **GitHub
Release assets**. Grab the latest from the
[releases page](https://github.com/nawat-john/air-hockey-master/releases), or
with the GitHub CLI:

```bash
gh release download v0.1                       # fetch runs/sac_final.zip
gh release create v0.2 runs/sac_final.zip \    # publish a new one
  --title "v0.2" --notes "trained agent"
```

## Current status

The agent now **beats the scripted predictor 59%** (118W/79L/3D over 200 eps),
up from 19.5% on the first run. Two things got it there: a higher update-to-data
ratio (`--gradient-steps 4`, which lifted it to 30.5%) and then another 1.5M
self-play steps from that checkpoint (→ 59%). This `runs/sac_final.zip` is the
one shipped to the web game (`RL (trained)` difficulty).

Known weakness: **own goals** (~51/200 eps) are now the dominant loss mode — the
policy plays aggressively and sometimes scores on itself. A dense
shoot-backward penalty in `AirHockeyEnv._reward` didn't dent it; reducing own
goals (and pushing win-rate past 59%) is the next training target.
