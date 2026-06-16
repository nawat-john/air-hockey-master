"""Train the air-hockey agent with SAC (plan §5, §8).

Curriculum phases (run them in order, each continuing from the last):

    python scripts/train_sac.py --phase defend   --timesteps 150000
    python scripts/train_sac.py --phase attack   --timesteps 200000 --load runs/defend.zip
    python scripts/train_sac.py --phase full     --timesteps 400000 --load runs/attack.zip
    python scripts/train_sac.py --phase selfplay --timesteps 800000 --load runs/full.zip

`selfplay` keeps a checkpoint pool and samples opponents from it (PFSP-lite),
mixing in the scripted predictor so the agent never forgets the basics.
TensorBoard logs land in runs/tb.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from airhockey.config import TableConfig
from airhockey.env import AirHockeyEnv

RUNS = Path(__file__).resolve().parents[1] / "runs"
POOL = RUNS / "pool"

PHASES = {
    # phase    -> (opponent, serve_mode,    max_seconds, randomize)
    "defend":   ("still",    "to_agent",    12.0, False),
    "attack":   ("scripted", "to_opponent", 15.0, False),
    "full":     ("scripted", "random",      20.0, False),
    "selfplay": ("scripted", "random",      20.0, True),
}


def make_env(cfg, opponent, serve_mode, max_seconds, randomize):
    def _thunk():
        return AirHockeyEnv(cfg=cfg, opponent=opponent, serve_mode=serve_mode,
                            max_episode_seconds=max_seconds, shaping=True,
                            randomize=randomize)
    return _thunk


class SelfPlayCallback(BaseCallback):
    """Periodically snapshot the policy into a pool and resample opponents.

    Each refresh, every env draws an opponent: the scripted predictor with
    probability ``p_scripted``, otherwise a random frozen checkpoint from the
    pool (a lightweight PFSP — uniform over recent-enough versions).
    """

    def __init__(self, cfg, save_every=40000, p_scripted=0.3, verbose=1):
        super().__init__(verbose)
        self.cfg = cfg
        self.save_every = save_every
        self.p_scripted = p_scripted
        self._last = 0
        POOL.mkdir(parents=True, exist_ok=True)

    def _snapshot(self):
        path = POOL / f"ckpt_{self.num_timesteps:08d}.zip"
        self.model.save(path)
        if self.verbose:
            print(f"[selfplay] snapshot -> {path.name}")

    def _resample(self):
        from airhockey.opponents import ScriptedPredictor, PolicyOpponent
        ckpts = sorted(POOL.glob("ckpt_*.zip"))
        for env_idx in range(self.training_env.num_envs):
            if not ckpts or np.random.rand() < self.p_scripted:
                opp = ScriptedPredictor(self.cfg)
            else:
                ck = np.random.choice(ckpts)
                opp = PolicyOpponent(SAC.load(ck, device="cpu"), self.cfg)
            self.training_env.env_method("set_opponent", opp, indices=[env_idx])

    def _on_training_start(self) -> None:
        self._snapshot()
        self._resample()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.save_every:
            self._last = self.num_timesteps
            self._snapshot()
            self._resample()
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), default="full")
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--load", default=None, help="checkpoint to continue from")
    ap.add_argument("--out", default=None, help="output .zip (default runs/<phase>.zip)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gradient-steps", type=int, default=1,
                    help="SAC gradient steps per rollout; raise for a higher "
                         "update-to-data ratio (-1 = n_envs => UTD~1)")
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    cfg = TableConfig()
    opponent, serve_mode, max_seconds, randomize = PHASES[args.phase]

    venv = DummyVecEnv([make_env(cfg, opponent, serve_mode, max_seconds, randomize)
                        for _ in range(args.n_envs)])

    if args.load:
        print(f"loading {args.load}")
        model = SAC.load(args.load, env=venv, device=args.device)
        # The replay buffer isn't saved with the model, so warm it up again.
        model.gradient_steps = args.gradient_steps
        model.learning_starts = 5000
    else:
        model = SAC(
            "MlpPolicy", venv,
            learning_rate=3e-4, batch_size=256, buffer_size=1_000_000,
            gamma=0.99, tau=0.005, ent_coef="auto", train_freq=1,
            gradient_steps=args.gradient_steps, learning_starts=5000,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(RUNS / "tb"), device=args.device, verbose=1,
        )

    callback = SelfPlayCallback(cfg) if args.phase == "selfplay" else None
    model.learn(total_timesteps=args.timesteps, callback=callback,
                tb_log_name=args.phase, reset_num_timesteps=args.load is None,
                progress_bar=True)

    out = Path(args.out) if args.out else RUNS / f"{args.phase}.zip"
    model.save(out)
    model.save(RUNS / "sac_final.zip")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
