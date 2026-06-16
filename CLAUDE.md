# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A 2D air-hockey simulator plus a hybrid (analytic physics + RL) bot that decides
at **10 Hz**. The full design rationale lives in `air_hockey_bot_plan.md` (Thai)
— read it before making architectural changes; the code deliberately mirrors its
sections (referenced as "plan §N" in docstrings).

## Environment / commands

This project uses the self-contained Python on `D:` — **always invoke it by full
path**, there is no activated venv in CI:

```
D:\Code\.venv\Scripts\python.exe
```

- Tests (run before training — see plan §10): `D:\Code\.venv\Scripts\python.exe -m pytest tests -q`
- Single test: `... -m pytest tests/test_physics.py::test_no_tunnelling_fast_puck -q`
- Headless runs (no display, e.g. tests/training): prefix `SDL_VIDEODRIVER=dummy` (pygame is imported lazily but `Renderer` needs it).
- Train: `... scripts\train_sac.py --phase {defend,attack,full,selfplay} --timesteps N [--load prev.zip]`
- Evaluate: `... scripts\evaluate.py --model runs\sac_final.zip --episodes 200`
- Play vs bot: `... scripts\play_human.py [--model runs\sac_final.zip]`

**torch must stay the `+cu124` CUDA build.** Installing/upgrading SB3 or other
packages can silently pull the CPU wheel and break GPU training. Verify with
`python -c "import torch; print(torch.cuda.is_available())"` and restore via the
`--index-url https://download.pytorch.org/whl/cu124` install if needed.

## Architecture (the big picture)

Three layers, intentionally separated so each can be tested alone:

1. **`physics.py` — `AirHockeyPhysics`** is *role-agnostic*: it owns mallet 0
   (left), mallet 1 (right), and the puck, and advances one substep given each
   mallet's target velocity. It knows nothing about agents, observations, or
   rewards. It runs at `physics_hz` (200) with positional-correction collisions
   (no tunnelling) and enforces mallet max speed **and** max accel (no teleport).
   `step()` returns a `GOAL_*` event code.

2. **`predictor.py` — `TrajectoryPredictor`** is pure, stateless analytic
   kinematics. The key trick is **mirror unfolding** (`_fold`): reflecting the
   table across its side walls turns a bouncing path into a straight line, giving
   an analytic `intercept(puck, x_line)` and `aim_point` for direct/bank shots.
   It ignores friction and mallets by design — tests that compare it to the sim
   must disable friction, set `wall_e=1.0`, and move mallets out of the path.

3. **`env.py` — `AirHockeyEnv`** (Gymnasium) ties them together. The agent is
   **always the left mallet, defends x=0, attacks +x** (no per-agent mirroring).
   Each `step()` runs `cfg.substeps` physics substeps; the opponent picks its
   target once per decision. `build_observation()` is a module-level function
   (16-d, left-attacker frame) **shared** by the env and by mirrored self-play
   opponents so both see an identical layout — keep them in sync if you change
   the obs.

### Coordinate convention & mirroring

Left goal = x=0 (agent), right goal = x=L (opponent). To let one left-trained
policy also drive the *right* mallet for self-play, `opponents.mirror_state`
reflects the world in x (`x→L−x`, `vx→−vx`) so the right player looks like a left
attacker; `PolicyOpponent` un-mirrors the resulting action. Any change to the
coordinate system must update both `mirror_state` and `build_observation`.

### Training flow (`train_sac.py`)

Curriculum phases in `PHASES` are meant to be run in sequence, each `--load`ing
the previous `.zip`. `selfplay` adds `SelfPlayCallback`: it snapshots the policy
into `runs/pool/` and, per env, resamples an opponent — scripted with prob
`p_scripted`, else a random frozen pool checkpoint (PFSP-lite). Self-play uses
`DummyVecEnv` (not Subproc) because opponents hold live SB3 models that don't
pickle cleanly across processes.

## Conventions

- Config is centralized in `config.TableConfig` (a frozen dataclass, SI units).
  Don't hardcode geometry/dynamics elsewhere — derive from `cfg`.
- New reward terms go in `AirHockeyEnv._reward`; gate shaping behind `self.shaping`
  and add sparse terms first, then shaping, watching for reward hacking (plan §10).
- `runs/`, `*.zip`, `*.gif` are gitignored — they're training artifacts.
