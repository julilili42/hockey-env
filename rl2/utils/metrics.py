import numpy as np
import os


class MetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.winrates = []

    def log_episode(self, reward):
        self.episode_rewards.append(reward)

    def log_update(self, actor_loss, critic_loss):
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)

    def log_eval(self, winrate):
        self.winrates.append(winrate)

    def moving_avg(self, window=100):
        if len(self.episode_rewards) < window:
            return np.array([])
        return np.convolve(
            self.episode_rewards,
            np.ones(window) / window,
            mode="valid"
        )

    def avg_reward(self, window=100):
        if len(self.episode_rewards) < window:
            return np.mean(self.episode_rewards)
        return self.moving_avg(window)[-1]


def save_metrics(metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, "metrics.npz"),
        rewards=metrics.episode_rewards,
        actor_losses=metrics.actor_losses,
        critic_losses=metrics.critic_losses,
        winrates=metrics.winrates,
    )
