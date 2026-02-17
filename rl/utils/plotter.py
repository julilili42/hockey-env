import matplotlib.pyplot as plt
import numpy as np
import os
import json
from types import SimpleNamespace


plt.rcParams.update({
    # Figure
    "figure.figsize": (5.5, 3.5),   
    "figure.dpi": 300,
    "savefig.dpi": 300,

    # Fonts
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    # Lines
    "lines.linewidth": 1.5,

    # Grid
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",

    # Remove top/right border
    "axes.spines.top": False,
    "axes.spines.right": False,
})


class MetricsPlotter:
    def __init__(self, metrics):
        self.metrics = metrics

    def _save_fig(self, path):
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def save_rewards(self, save_dir, window=100):
        rewards = np.array(self.metrics.episode_rewards)
        if len(rewards) == 0:
            return

        plt.figure()

        plt.plot(rewards,
                color="tab:blue",
                alpha=0.2,
                label="Episode reward")

        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode="valid")

            std = np.array([
                np.std(rewards[i-window:i])
                for i in range(window, len(rewards)+1)
            ])

            x = np.arange(window-1, len(rewards))

            plt.plot(x, ma,
                    color="tab:blue",
                    linewidth=2.2,
                    label=f"{window}-episode moving average")

            plt.fill_between(x,
                            ma - std,
                            ma + std,
                            color="tab:blue",
                            alpha=0.15)

        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Training Performance")
        plt.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "rewards.pdf"))
        plt.close()



    def save_losses(self, save_dir):
        if len(self.metrics.critic_losses) == 0:
            return

        critic = np.array(self.metrics.critic_losses)
        actor = np.array(self.metrics.actor_losses)

        fig, ax1 = plt.subplots()

        ax1.plot(critic,
                color="tab:blue",
                label="Critic loss")
        ax1.set_xlabel("Update step")
        ax1.set_ylabel("Critic loss")

        if len(actor) > 0:
            ax2 = ax1.twinx()
            ax2.plot(actor,
                    linestyle="--",
                    color="tab:orange",
                    label="Actor loss")
            ax2.set_ylabel("Actor loss")

        ax1.set_title("Optimization Loss")

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, "losses.pdf"))
        plt.close()




    def save_winrate(self, save_dir):
        strong = np.array(self.metrics.winrate_strong)
        weak = np.array(self.metrics.winrate_weak)
        min_wr = np.array(self.metrics.winrate_min)

        if len(min_wr) == 0:
            return

        plt.figure()
        plt.plot(strong, label="Strong")
        plt.plot(weak, label="Weak")
        plt.plot(min_wr, label="Min", linewidth=2)

        plt.axhline(0.5, linestyle="--", color="gray", label="Random")
        plt.ylim(0, 1)

        plt.xlabel("Evaluation Step")
        plt.ylabel("Winrate")
        plt.title("Evaluation Performance")
        plt.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "winrate.pdf"))
        plt.close()





    def save_combined(
    self,
    save_dir,
    window=100,
    eval_interval=200,
    show="weak"  # "weak" | "strong" | "both" | "min"
    ):
        rewards = np.array(self.metrics.episode_rewards)
        weak_wr = np.array(self.metrics.winrate_weak)
        strong_wr = np.array(self.metrics.winrate_strong)
        min_wr = np.array(self.metrics.winrate_min)

        if len(rewards) < window:
            return

        fig, ax1 = plt.subplots()

        # ---- Moving average + std ----
        ma = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid"
        )

        std = np.array([
            np.std(rewards[i - window:i])
            for i in range(window, len(rewards) + 1)
        ])

        x_reward = np.arange(window - 1, len(rewards))

        ax1.plot(
            x_reward,
            ma,
            color="#1f77b4",
            linewidth=1.8,
            label=f"Return ({window}-episode MA)"
        )

        ax1.fill_between(
            x_reward,
            ma - std,
            ma + std,
            color="#1f77b4",
            alpha=0.12
        )

        ax1.set_xlabel("Episodes")
        ax1.set_ylabel("Average Return")

        # ---- Winrate axis ----
        ax2 = ax1.twinx()

        if len(weak_wr) > 0:
            x_eval = np.arange(
                eval_interval,
                eval_interval * (len(weak_wr) + 1),
                eval_interval
            )
        else:
            x_eval = None

        plotted = []

        if show in ["weak", "both"] and len(weak_wr) > 0:
            ax2.plot(
                x_eval,
                weak_wr,
                color="#d62728",
                marker="o",
                markersize=3,
                linestyle="none",
                label="Winrate (Weak)"
            )
            plotted.append(weak_wr)

        if show in ["strong", "both"] and len(strong_wr) > 0:
            ax2.plot(
                x_eval,
                strong_wr,
                color="#2ca02c",
                marker="s",
                markersize=3,
                linestyle="none",
                label="Winrate (Strong)"
            )
            plotted.append(strong_wr)

        if show == "min" and len(min_wr) > 0:
            ax2.plot(
                x_eval,
                min_wr,
                color="black",
                marker="d",
                markersize=3,
                linestyle="none",
                label="Winrate (Min)"
            )
            plotted.append(min_wr)

        ax2.axhline(
            0.5,
            linestyle="--",
            linewidth=1,
            color="gray",
            alpha=0.6,
            label="Random"
        )

        # ---- Dynamic y-limit with small headroom ----
        if plotted:
            max_val = max(arr.max() for arr in plotted)
            upper = min(1.02, max_val + 0.02)
            ax2.set_ylim(0.0, upper)
        else:
            ax2.set_ylim(0.0, 1.02)

        ax2.set_ylabel("Winrate")

        # ---- Combined legend ----
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            frameon=False,
            loc="lower right"
        )

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, "combined.pdf"))
        plt.close()








    def save_opponents(self, save_dir):
        history = self.metrics.opponent_history
        if len(history) == 0:
            return

        episodes = [h["episode"] for h in history]
        strong = [h["strong"] for h in history]
        weak = [h["weak"] for h in history]
        sp = [h["self_play"] for h in history]
        sp_prob = [h["self_play_prob"] for h in history]

        plt.figure()

        # Stacked area plot für echte Verteilung
        plt.stackplot(
            episodes,
            strong,
            weak,
            sp,
            labels=["Strong", "Weak", "SelfPlay"],
            alpha=0.7
        )

        # self_play_prob als Linie drüber
        plt.plot(
            episodes,
            sp_prob,
            linestyle="--",
            linewidth=2,
            label="SelfPlay Prob",
            color="black"
        )

        plt.xlabel("Episode")
        plt.ylabel("Ratio / Probability")
        plt.title("Opponent Distribution Over Episodes")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "opponents.pdf"))
        plt.close()


    def save_eval_rewards(self, save_dir):
        strong_r = np.array(self.metrics.reward_strong)
        weak_r = np.array(self.metrics.reward_weak)

        if len(strong_r) == 0 and len(weak_r) == 0:
            return

        plt.figure()

        if len(strong_r) > 0:
            plt.plot(strong_r, label="Reward Strong")

        if len(weak_r) > 0:
            plt.plot(weak_r, label="Reward Weak")

        plt.xlabel("Evaluation Step")
        plt.ylabel("Average Episode Return")
        plt.title("Evaluation Rewards")
        plt.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "eval_rewards.pdf"))
        plt.close()



    @classmethod
    def from_json(cls, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        metrics = SimpleNamespace(**data)

        # Map plural names from JSON to singular names used in plotter
        mapping = {
            "winrate_strong": "winrates_strong",
            "winrate_weak": "winrates_weak",
            "winrate_min": "winrates_min"
        }

        for new_name, old_name in mapping.items():
            if hasattr(metrics, old_name):
                setattr(metrics, new_name, getattr(metrics, old_name))
            else:
                setattr(metrics, new_name, [])

        # Ensure optional fields exist
        optional_fields = [
            "episode_rewards",
            "actor_losses",
            "critic_losses",
            "opponent_history",
            "reward_strong",
            "reward_weak"
        ]

        for field in optional_fields:
            if not hasattr(metrics, field):
                setattr(metrics, field, [])

        # Add moving average
        def moving_avg(window):
            rewards = np.array(metrics.episode_rewards)
            if len(rewards) < window:
                return []
            return np.convolve(
                rewards,
                np.ones(window) / window,
                mode="valid"
            )

        metrics.moving_avg = moving_avg

        return cls(metrics)



    def save_all(self, save_dir, window=100, show="weak"):
        os.makedirs(save_dir, exist_ok=True)

        self.save_rewards(save_dir, window)
        self.save_losses(save_dir)
        self.save_winrate(save_dir)
        self.save_eval_rewards(save_dir)
        self.save_combined(save_dir, window=window, show=show)
        self.save_opponents(save_dir)


