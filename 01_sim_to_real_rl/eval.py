"""Headless eval of a saved policy.

    python3 eval.py [policy_name_without_zip] [ppo|sac] [n_seeds] [damping_max]

Per seed it counts steps within 0.2 rad of upright, the longest continuous hold, and whether the
pole is still balanced at the end of the episode (last 50 steps all within 0.2 rad).
damping_max sets the evaluation range U(0.001, damping_max); default 0.02 (the solvable range),
use 0.05 for the original wide range. With 50+ seeds it also prints balanced-at-end by sampled
hinge damping, because that is what decided success on the wide range.
"""
import sys
import numpy as np
import mujoco
from stable_baselines3 import PPO, SAC
from balance_env import BalanceBotEnv, policy_path

name = sys.argv[1] if len(sys.argv) > 1 else "mujoco_sac_feasible"
algo = (sys.argv[2] if len(sys.argv) > 2 else "sac").lower()
n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 8
damping_max = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02

model = (SAC if algo == "sac" else PPO).load(policy_path(name), device="cpu")
env = BalanceBotEnv(damping_range=(0.001, damping_max))
hinge = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")

ups, damps, balanced = [], [], []
for s in range(n_seeds):
    obs, _ = env.reset(seed=s)
    damping = env.model.dof_damping[hinge]  # sampled by reset()
    n = up = run = best = 0
    first = None
    errs = []
    d = t = False
    while not (d or t):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, d, t, _ = env.step(np.array(a).flatten())
        n += 1
        err = abs(np.arctan2(np.sin(env.data.qpos[1]), np.cos(env.data.qpos[1])))
        errs.append(err)
        if err < 0.2:
            up += 1
            run += 1
            best = max(best, run)
            if first is None:
                first = n
        else:
            run = 0
    ok = all(e < 0.2 for e in errs[-50:])
    ups.append(up)
    damps.append(damping)
    balanced.append(ok)
    if n_seeds <= 20:
        print(f"seed={s} len={n} steps_upright={up} longest_hold={best} first={first} "
              f"balanced_at_end={ok} damping={damping:.3f} final_x={env.data.qpos[0]:+.2f} "
              f"ended={'cart-limit' if d else 'time'}")

balanced, damps = np.array(balanced), np.array(damps)
print(f"damping range U(0.001, {damping_max}); mean steps_upright {np.mean(ups):.1f}/500")
print(f"balanced at end: {balanced.sum()}/{n_seeds} = {balanced.mean():.2f}")
if n_seeds >= 50:
    print("balanced at end by hinge damping:")
    for lo in np.arange(0.0, damping_max, 0.005):
        m = (damps >= lo) & (damps < lo + 0.005)
        if m.any():
            print(f"  damping {lo:.3f}-{lo + 0.005:.3f}: {balanced[m].sum():3d}/{m.sum():3d} = {balanced[m].mean():.2f}")
