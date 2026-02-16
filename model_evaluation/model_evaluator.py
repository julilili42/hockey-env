# rl/eval_models.py

import argparse
import csv
import glob
import os
import re

import gymnasium as gym
import numpy as np
import torch

import hockey.hockey_env
from rl.td3.agent import TD3Agent
from rl.td3.config import TD3Config


class ModelEvaluator:

    def __init__(self, patterns, episodes, seed, group_regex, out_dir):
        self.patterns = patterns
        self.episodes = episodes
        self.seed = seed
        self.group_regex = group_regex
        self.out_dir = out_dir

    def run(self):
        model_paths = self._find_models()
        if not model_paths:
            raise SystemExit("No checkpoints found.")

        results = []

        for p in model_paths:
            label = self._make_label(p)

            wr_w, ret_w = self._eval_once(p, True)
            wr_s, ret_s = self._eval_once(p, False)

            results.append(dict(
                label=label,
                path=p,
                wr_weak=wr_w,
                wr_strong=wr_s,
                ret_weak=ret_w,
                ret_strong=ret_s,
                episodes=self.episodes,
            ))

            print(f"{label:30s} | WR_w={wr_w:.3f} WR_s={wr_s:.3f}")

        agg = self._aggregate(results)

        raw_csv = os.path.join(self.out_dir, "results_raw.csv")
        grouped_csv = os.path.join(self.out_dir, "results_grouped.csv")
        latex_tbl = os.path.join(self.out_dir, "table_final_eval.tex")

        self._write_csv_raw(raw_csv, results)
        self._write_csv_grouped(grouped_csv, agg)
        self._write_latex_table(latex_tbl, agg)

        print("\nSaved:")
        print(" ", raw_csv)
        print(" ", grouped_csv)
        print(" ", latex_tbl)

    def _find_models(self):
        out = []
        for p in self.patterns:
            out.extend(glob.glob(p, recursive=True))
        return sorted(set(x for x in out if os.path.isfile(x)))

    def _make_label(self, path):
        norm = path.replace("\\", "/")
        if self.group_regex:
            m = re.search(self.group_regex, norm)
            if m:
                return m.group(1) if m.groups() else m.group(0)
        return os.path.basename(os.path.dirname(path))

    def _eval_once(self, model_path, opponent_weak):
        env = gym.make("Hockey-One-v0", weak_opponent=opponent_weak)

        cfg = TD3Config()
        agent = TD3Agent(env=env, config=cfg, h=256, seed=self.seed)
        agent.load(model_path)
        agent.policy.eval()

        wins = []
        returns = []

        with torch.no_grad():
            for i in range(self.episodes):
                obs, _ = env.reset(seed=self.seed + i)
                done = False
                trunc = False
                ep_ret = 0.0

                while not (done or trunc):
                    a = agent.get_action(obs, noise=False, eval_mode=True)
                    obs, r, done, trunc, info = env.step(a)
                    ep_ret += float(r)

                wins.append(1 if info.get("winner", 0) == 1 else 0)
                returns.append(ep_ret)

        env.close()
        return float(np.mean(wins)), float(np.mean(returns))

    def _aggregate(self, results):
        grouped = {}
        for r in results:
            grouped.setdefault(r["label"], []).append(r)

        out = {}
        for label, items in grouped.items():
            wr_w = np.array([x["wr_weak"] for x in items])
            wr_s = np.array([x["wr_strong"] for x in items])
            rt_w = np.array([x["ret_weak"] for x in items])
            rt_s = np.array([x["ret_strong"] for x in items])

            out[label] = dict(
                n_models=len(items),
                wr_weak_mean=wr_w.mean(),
                wr_weak_std=wr_w.std(ddof=1) if len(items) > 1 else 0.0,
                wr_strong_mean=wr_s.mean(),
                wr_strong_std=wr_s.std(ddof=1) if len(items) > 1 else 0.0,
                ret_weak_mean=rt_w.mean(),
                ret_weak_std=rt_w.std(ddof=1) if len(items) > 1 else 0.0,
                ret_strong_mean=rt_s.mean(),
                ret_strong_std=rt_s.std(ddof=1) if len(items) > 1 else 0.0,
            )

        return out

    def _write_csv_raw(self, path, results):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "model_path", "wr_weak", "wr_strong",
                        "ret_weak", "ret_strong", "eval_episodes"])
            for r in results:
                w.writerow([
                    r["label"],
                    r["path"],
                    f"{r['wr_weak']:.6f}",
                    f"{r['wr_strong']:.6f}",
                    f"{r['ret_weak']:.6f}",
                    f"{r['ret_strong']:.6f}",
                    r["episodes"],
                ])

    def _write_csv_grouped(self, path, agg):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "variant", "n_models",
                "wr_weak_mean", "wr_weak_std",
                "wr_strong_mean", "wr_strong_std",
                "ret_weak_mean", "ret_weak_std",
                "ret_strong_mean", "ret_strong_std",
            ])
            for label in sorted(agg.keys()):
                a = agg[label]
                w.writerow([
                    label,
                    a["n_models"],
                    f"{a['wr_weak_mean']:.6f}", f"{a['wr_weak_std']:.6f}",
                    f"{a['wr_strong_mean']:.6f}", f"{a['wr_strong_std']:.6f}",
                    f"{a['ret_weak_mean']:.6f}", f"{a['ret_weak_std']:.6f}",
                    f"{a['ret_strong_mean']:.6f}", f"{a['ret_strong_std']:.6f}",
                ])

    def _write_latex_table(self, path, agg):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        labels = sorted(agg.keys())
        best_label = None
        best_score = -1e9

        for lab in labels:
            a = agg[lab]
            score = min(a["wr_weak_mean"], a["wr_strong_mean"])
            if score > best_score:
                best_score = score
                best_label = lab

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{lcccc}",
            "\\hline",
            "Variant & WR Weak & WR Strong & Return Weak & Return Strong \\\\",
            "\\hline",
        ]

        for lab in labels:
            a = agg[lab]
            is_best = lab == best_label

            wr_w = f"{100*a['wr_weak_mean']:.2f} $\\pm$ {100*a['wr_weak_std']:.2f}"
            wr_s = f"{100*a['wr_strong_mean']:.2f} $\\pm$ {100*a['wr_strong_std']:.2f}"
            rt_w = f"{a['ret_weak_mean']:.2f} $\\pm$ {a['ret_weak_std']:.2f}"
            rt_s = f"{a['ret_strong_mean']:.2f} $\\pm$ {a['ret_strong_std']:.2f}"

            if is_best:
                lab = f"\\textbf{{{lab}}}"
                wr_w = f"\\textbf{{{wr_w}}}"
                wr_s = f"\\textbf{{{wr_s}}}"
                rt_w = f"\\textbf{{{rt_w}}}"
                rt_s = f"\\textbf{{{rt_s}}}"

            lines.append(f"{lab} & {wr_w}\\% & {wr_s}\\% & {rt_w} & {rt_s} \\\\")

        lines += [
            "\\hline",
            "\\end{tabular}",
            "\\caption{Final evaluation (mean $\\pm$ std across seeds).}",
            "\\label{tab:final_eval}",
            "\\end{table}",
        ]

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "runs/**/models/td3_best.pt",
        "pretrained/**/models/td3_best.pt",
    ])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--group_regex", type=str, default=None)
    ap.add_argument("--out_dir", type=str,
                    default="runs/comparisons/final_eval")
    args = ap.parse_args()

    evaluator = ModelEvaluator(
        patterns=args.models,
        episodes=args.episodes,
        seed=args.seed,
        group_regex=args.group_regex,
        out_dir=args.out_dir,
    )

    evaluator.run()


if __name__ == "__main__":
    main()
