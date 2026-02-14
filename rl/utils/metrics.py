import numpy as np
import os
import json


class MetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.winrates = []
        self.opponent_history = []


    def log_episode(self, reward):
        self.episode_rewards.append(reward)

    def log_update(self, actor_loss, critic_loss):
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)

    def log_eval(self, winrate):
        self.winrates.append(winrate)

    def log_opponent_dist(self, episode, strong, weak, self_play, self_play_prob):
        self.opponent_history.append({
            "episode": episode,
            "strong": strong,
            "weak": weak,
            "self_play": self_play,
            "self_play_prob": self_play_prob,
        })



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

    data = {
        "episode_rewards": metrics.episode_rewards,
        "actor_losses": metrics.actor_losses,
        "critic_losses": metrics.critic_losses,
        "winrates": metrics.winrates,
        "opponent_history": metrics.opponent_history,
    }

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(data, f, indent=4)
