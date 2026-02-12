import matplotlib.pyplot as plt
import numpy as np
import os


class MetricsPlotter:
    def __init__(self, metrics):
        self.metrics = metrics

    def _save_fig(self, path):
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def save_rewards(self, save_dir, window=100):
        rewards = self.metrics.episode_rewards
        if len(rewards) == 0:
            return

        plt.figure(figsize=(6, 4))
        plt.plot(rewards, alpha=0.3, label="Episode reward")

        if len(rewards) >= window:
            ma = self.metrics.moving_avg(window)
            plt.plot(
                range(window - 1, len(rewards)),
                ma,
                label=f"{window}-episode moving avg"
            )

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()

        self._save_fig(os.path.join(save_dir, "rewards.png"))

    def save_losses(self, save_dir):
        if len(self.metrics.critic_losses) == 0:
            return

        fig, ax1 = plt.subplots(figsize=(6, 4))

        # Critic (linke Achse)
        ax1.set_xlabel("Update step")
        ax1.set_ylabel("Critic loss")
        ax1.plot(self.metrics.critic_losses, label="Critic loss")
        ax1.tick_params(axis='y')

        # Actor (rechte Achse)
        if len(self.metrics.actor_losses) > 0:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Actor loss")
            ax2.plot(self.metrics.actor_losses, linestyle="--", label="Actor loss")
            ax2.tick_params(axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "losses.png"))
        plt.close()


    def save_winrate(self, save_dir):
        if len(self.metrics.winrates) == 0:
            return

        plt.figure(figsize=(6, 4))
        plt.plot(self.metrics.winrates)
        plt.axhline(0.55, linestyle="--")
        plt.xlabel("Evaluation step")
        plt.ylabel("Winrate")

        self._save_fig(os.path.join(save_dir, "winrate.png"))

    def save_all(self, save_dir, window=100):
        os.makedirs(save_dir, exist_ok=True)

        self.save_rewards(save_dir, window)
        self.save_losses(save_dir)
        self.save_winrate(save_dir)
