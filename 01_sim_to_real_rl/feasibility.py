"""Is swing-up physically possible at a given hinge damping?  (cross-entropy search over open-loop cart forces)

    python3 feasibility.py [--damping 0.005 0.02 0.03 0.05] [--force 10] [--track 3] [--seconds 20] [--seeds 2]

For each damping it searches open-loop force sequences for the highest point the pole reaches while the cart
stays inside +-track metres. "max cos(phi)" = +1 means the pole reached upright; anything below ~0.98 means the
search could not get it there. A search is not a proof, but it matches the physics estimate (the cart must supply
m*lc*a*w >= c*w^2 to beat damping c) and the point where the learned SAC policy also stops working.

Findings with the env's limits (10 N, +-3 m, 20 s): reaches the top up to damping 0.02, fails from ~0.022 up.
At damping 0.04-0.05 a longer track or more time changes nothing; doubling the force to 20 N does, so motor force binds.
"""
import argparse
import time
import numpy as np
import mujoco
from mujoco import rollout
from balance_env import BalanceBotEnv

p = argparse.ArgumentParser()
p.add_argument("--damping", type=float, nargs="+", default=[0.005, 0.02, 0.03, 0.04, 0.05])
p.add_argument("--force", type=float, default=10.0, help="motor force limit in N (env: 10)")
p.add_argument("--track", type=float, default=3.0, help="cart limit in m (env: 3)")
p.add_argument("--seconds", type=float, default=20.0, help="horizon (env: 20 s = 500 steps)")
p.add_argument("--seeds", type=int, default=2)
p.add_argument("--iters", type=int, default=50)
args = p.parse_args()

SUB = 16                                  # physics steps per knot (0.16 s)
K = int(round(args.seconds / (SUB * 0.01)))
H = K * SUB
N = 512                                   # candidates per CEM iteration

model = BalanceBotEnv().model
model.actuator_ctrlrange[0] = [-args.force, args.force]
hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
datas = [mujoco.MjData(model) for _ in range(16)]
init = np.array([[0.0, 0.0, np.pi, 0.0, 0.0]])  # [time, x, phi (pi = hanging), xdot, phidot]


def fitness(U):
    """Highest cos(phi) reached before the cart leaves the track (the env would terminate there)."""
    ctrl = np.repeat(U * args.force, SUB, axis=1)[:, :, None]
    st, _ = rollout.rollout(model, datas, init, ctrl)
    x, phi = st[:, :, 1], st[:, :, 2]
    viol = np.abs(x) > args.track
    first = np.where(viol.any(1), viol.argmax(1), H)
    ok = np.arange(H)[None, :] < first[:, None]
    return np.where(ok, np.cos(phi), -1.0).max(1)


def search(damping, seed):
    model.dof_damping[hinge] = damping
    rng = np.random.default_rng(seed)
    mu, sd = np.zeros(K), np.full(K, 0.7)
    best = -1.0
    for _ in range(args.iters):
        U = np.clip(mu + sd * rng.standard_normal((N, K)), -1, 1)
        f = fitness(U)
        elite = np.argsort(f)[-N // 10:]
        mu = 0.7 * U[elite].mean(0) + 0.3 * mu
        sd = U[elite].std(0) + 0.03
        best = max(best, f.max())
    return best


print(f"force {args.force} N, track +-{args.track} m, horizon {args.seconds} s")
for c in args.damping:
    t0 = time.time()
    best = max(search(c, s) for s in range(args.seeds))
    print(f"damping={c:.3f}: max cos(phi) = {best:+.3f}  {'reaches upright' if best > 0.98 else 'cannot reach upright'}"
          f"  [{time.time() - t0:.0f}s]", flush=True)
