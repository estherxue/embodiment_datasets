"""Fill the result sections of embodiment_datasets/LIBERO/EVAL_NOTES.md from downloaded Kaggle outputs.

Usage: python merge_notes.py <run_dir> [<run_dir> ...]   (later runs override earlier ones per task)
"""
import glob, json, math, os, sys

here = os.path.dirname(os.path.abspath(__file__))
notes_path = os.path.join(here, "..", "..", "LIBERO", "EVAL_NOTES.md")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_runs(run_dirs):
    replay, protocol, summaries = {}, None, {}
    for rd in run_dirs:
        for f in glob.glob(os.path.join(rd, "replay", "libero_spatial_task*_replay.json")):
            d = json.load(open(f))
            if d["records"]:
                replay[d["task_id"]] = (d, os.path.basename(rd))
        sp = os.path.join(rd, "summary.json")
        if os.path.exists(sp):
            sj = json.load(open(sp))
            summaries[os.path.basename(rd)] = sj
            pro = sj.get("results", {}).get("protocol", {})
            if any(t.get("eval_summary") for t in pro.get("per_task", [])):
                protocol = (pro, os.path.basename(rd), sj)
    return replay, protocol, summaries


def replay_section(replay):
    rows = []
    tot_n = tot_k = 0
    lines = ["| task | 来源 run | demo 数 | 成功 | 成功率 | Bernoulli std | SE | 95% CI (Wilson) | 结束时仍成功 | 平均成功步数 / demo 长度 | 状态漂移 mean / max | 秒/demo |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tid in sorted(replay):
        d, run = replay[tid]
        recs = d["records"]
        n, k = len(recs), sum(r["success"] for r in recs)
        p = k / n
        std = math.sqrt(p * (1 - p))
        se = std / math.sqrt(n)
        lo, hi = wilson(k, n)
        fin = sum(r["final_success"] for r in recs) / n
        steps = [r["success_step"] for r in recs if r["success"]]
        ms = sum(steps) / len(steps) if steps else float("nan")
        ml = sum(r["n_actions"] for r in recs) / n
        drifts = [r["max_state_drift"] for r in recs]
        sec = sum(r["seconds"] for r in recs) / n
        tot_n += n; tot_k += k
        rows.append(p)
        lines.append(f"| {tid} | {run} | {n} | {k} | {100*p:.0f}% | {std:.3f} | {se:.3f} | [{100*lo:.0f}%, {100*hi:.0f}%] | {100*fin:.0f}% | "
                     f"{ms:.1f} / {ml:.1f} | {sum(drifts)/n:.2f} / {max(drifts):.2f} | {sec:.0f} |")
    if rows:
        p = tot_k / tot_n
        lo, hi = wilson(tot_k, tot_n)
        m = sum(rows) / len(rows)
        sd = math.sqrt(sum((x - m) ** 2 for x in rows) / (len(rows) - 1)) if len(rows) > 1 else 0.0
        lines += ["", f"- 合计 {tot_k}/{tot_n} = **{100*p:.1f}%**（95% Wilson CI [{100*lo:.1f}%, {100*hi:.1f}%]）；任务间成功率均值 {100*m:.1f}%，跨任务标准差 {100*sd:.1f} 个百分点，"
                      f"任务级样本方差 {sd*sd:.4f}。",
                  "- 成功判定与评测协议相同：`env.step` 返回的 `done`（= `_check_success()`），某一步成功即记成功；“结束时仍成功”是回放完全部动作后再查一次 `check_success()`。",
                  "- 状态漂移 = 回放过程中 `env.get_sim_state()` 与数据集里记录的 `states` 的最大 L2 距离，用来说明开环回放的物理可复现程度（MuJoCo 非确定性 + 版本差异）。"]
    return "\n".join(lines)


def protocol_section(protocol):
    if protocol is None:
        return "（无结果）"
    pro, run, sj = protocol
    lines = [f"- 来源 run：{run}；checkpoint：{pro.get('note')}",
             f"- 最终配置：{pro.get('config_final') or pro.get('env_num_final')}；跳过的任务：{pro.get('skipped_tasks', [])}", "",
             "| task | 渲染后端 | episodes (env_num) | 实际步数 | max_steps | 成功数 | success_rate | rollout 秒 | 整段 wall 秒 | rc |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in pro["per_task"]:
        e = t.get("eval_summary") or {}
        lines.append(f"| {t['task_id']} | {t.get('backend', 'egl')} | {e.get('episodes', t['env_num'])} | {e.get('steps_run')} | {e.get('max_steps')} | {e.get('num_success')} | "
                     f"{e.get('success_rate')} | {e.get('seconds')} | {t.get('wall_s')} | {t['rc']} |")
    return "\n".join(lines)


def fill(s, key, body):
    a, b = f"<!-- {key} -->", f"<!-- /{key} -->"
    i, j = s.index(a) + len(a), s.index(b)
    return s[:i] + "\n" + body.strip("\n") + "\n" + s[j:]


if __name__ == "__main__":
    runs = [os.path.abspath(r) for r in sys.argv[1:]]
    replay, protocol, summaries = load_runs(runs)
    s = open(notes_path).read()
    s = fill(s, "KAGGLE_REPLAY", replay_section(replay))
    s = fill(s, "KAGGLE_PROTOCOL", protocol_section(protocol))
    open(notes_path, "w").write(s)
    print("replay tasks:", sorted(replay), "| protocol:", (protocol[1] if protocol else None), "| runs:", list(summaries))
