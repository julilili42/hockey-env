import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    # Figure
    "figure.figsize": (5.5, 3.5),   
    "figure.dpi": 300,
    "savefig.dpi": 300,

    # Fonts
    "font.family": "serif",
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
        winrates = np.array(self.metrics.winrates)
        if len(winrates) == 0:
            return

        plt.figure()

        plt.plot(winrates,
                marker="o",
                markersize=3,
                linewidth=1.8,
                color="tab:green",
                label="Winrate")

        # smoothing
        if len(winrates) >= 5:
            smooth = np.convolve(winrates, np.ones(5)/5, mode="valid")
            plt.plot(range(4, len(winrates)),
                    smooth,
                    linewidth=2.2,
                    color="black",
                    label="Smoothed")

        plt.axhline(0.5,
                    linestyle="--",
                    linewidth=1.2,
                    color="gray",
                    label="Random baseline")

        plt.ylim(0, 1)

        plt.xlabel("Evaluation Step")
        plt.ylabel("Winrate")
        plt.title("Evaluation Performance")
        plt.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "winrate.pdf"))
        plt.close()


    def save_combined(self, save_dir, window=100):
        rewards = self.metrics.moving_avg(window)
        winrates = np.array(self.metrics.winrates)

        if len(rewards) == 0 or len(winrates) == 0:
            return

        fig, ax1 = plt.subplots()

        ax1.plot(rewards,
                color="tab:blue",
                label="Reward (MA)")
        ax1.set_xlabel("Training progress")
        ax1.set_ylabel("Reward")

        ax2 = ax1.twinx()
        ax2.plot(
            np.linspace(0, len(rewards), len(winrates)),
            winrates,
            color="tab:red",
            label="Winrate"
        )
        ax2.set_ylabel("Winrate")

        ax1.set_title("Learning Progress")

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






    def save_all(self, save_dir, window=100):
        os.makedirs(save_dir, exist_ok=True)

        self.save_rewards(save_dir, window)
        self.save_losses(save_dir)
        self.save_winrate(save_dir)
        self.save_combined(save_dir)
        self.save_opponents(save_dir)

