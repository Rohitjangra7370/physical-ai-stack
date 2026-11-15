"""Train SAC on the swing-up env.

    python3 train_sac.py [steps] [--seed N] [--name OUT] [--damping-max X] [--threads N]

--damping-max sets the upper end of the per-episode hinge damping range U(0.001, X).
The default (0.02) is the physically solvable range; 0.05 reproduces the original wide range.
"""
import argparse
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from balance_env import BalanceBotEnv

p = argparse.ArgumentParser()
p.add_argument("steps", type=int, nargs="?", default=300_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--name", default="sac_run", help="output policy name (without .zip)")
p.add_argument("--damping-max", type=float, default=0.02)
p.add_argument("--threads", type=int, default=4, help="torch CPU threads")
args = p.parse_args()

torch.set_num_threads(args.threads)
env = Monitor(BalanceBotEnv(damping_range=(0.001, args.damping_max)))  # Monitor logs ep_rew_mean / ep_len_mean

# SAC defaults are tuned for this kind of task. ent_coef="auto" learns the exploration level
# (target entropy = -action_dim), so it keeps exploring until it has something better.
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    seed=args.seed,
    tensorboard_log="./sac_tensorboard/",
    device="cpu",
    learning_starts=5_000,  # random actions first to fill the replay buffer
    train_freq=1,
    gradient_steps=1,
)

ckpt = CheckpointCallback(save_freq=50_000, save_path="./checkpoints/", name_prefix=args.name)
print(f"training SAC for {args.steps} steps, damping U(0.001, {args.damping_max}), seed {args.seed}")
model.learn(total_timesteps=args.steps, callback=ckpt, tb_log_name=args.name)
model.save(args.name)
print(f"done SAC saved to {args.name}.zip")
