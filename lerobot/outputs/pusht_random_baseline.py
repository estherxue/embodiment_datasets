"""Random-action baseline on gym_pusht/PushT-v0 with the same seeds/episode length as lerobot-eval."""
import json, sys, time
import numpy as np, gymnasium as gym, gym_pusht  # noqa: F401
start_seed, n = int(sys.argv[1]), int(sys.argv[2])
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
               visualization_width=384, visualization_height=384, max_episode_steps=300, disable_env_checker=True)
rows = []; t = time.time()
for ep in range(n):
    seed = start_seed + ep
    obs, info = env.reset(seed=seed); rng = np.random.default_rng(seed)
    sum_r, max_r, succ, done = 0.0, 0.0, False, False
    while not done:
        obs, r, term, trunc, info = env.step(rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32))
        sum_r += r; max_r = max(max_r, r); succ = succ or bool(info.get("is_success", False)); done = term or trunc
    rows.append(dict(episode_ix=ep, sum_reward=sum_r, max_reward=max_r, success=succ, seed=seed))
agg = dict(avg_sum_reward=float(np.mean([x["sum_reward"] for x in rows])), avg_max_reward=float(np.mean([x["max_reward"] for x in rows])),
           pc_success=100 * float(np.mean([x["success"] for x in rows])), eval_s=time.time() - t, eval_ep_s=(time.time() - t) / n)
out = sys.argv[3]; json.dump(dict(per_episode=rows, aggregated=agg), open(out, "w"), indent=2)
print(agg)
