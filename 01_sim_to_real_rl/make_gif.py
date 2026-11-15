"""Render a policy rollout to a GIF without a display.

    python3 make_gif.py [policy_name] [seed] [out.gif] [steps]

Uses MuJoCo's offscreen renderer (EGL). One env step is 0.04 s of simulated time; every 2nd step is drawn,
so the GIF plays in real time at 12.5 fps.
"""
import os
import sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from PIL import Image
from stable_baselines3 import SAC
from balance_env import BalanceBotEnv, policy_path

name = sys.argv[1] if len(sys.argv) > 1 else "mujoco_sac_feasible"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12  # seed 12: damping 0.019, needs ~12 s of pumping
out = sys.argv[3] if len(sys.argv) > 3 else "swingup.gif"
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 500

model = SAC.load(policy_path(name), device="cpu")
env = BalanceBotEnv()
env.model.vis.headlight.ambient[:] = 0.45   # default scene is very dark
env.model.vis.headlight.diffuse[:] = 0.8
obs, _ = env.reset(seed=seed)

cam = mujoco.MjvCamera()
cam.lookat[:] = [0.0, 0.0, 0.85]
cam.distance = 3.0
cam.azimuth = 90
cam.elevation = -5

renderer = mujoco.Renderer(env.model, height=300, width=400)
frames = []
for k in range(steps):
    if k % 2 == 0:
        cam.lookat[0] = env.data.qpos[0]    # follow the cart
        renderer.update_scene(env.data, camera=cam)
        frames.append(Image.fromarray(renderer.render()))
    a, _ = model.predict(obs, deterministic=True)
    obs, r, d, t, _ = env.step(np.array(a).flatten())
    if d or t:
        break

frames[0].save(out, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
print(f"wrote {out}: {len(frames)} frames, {os.path.getsize(out) / 1e6:.2f} MB")
