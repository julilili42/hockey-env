import numpy as np
import torch

class Evaluator:
    def __init__(self, env, episodes=100, label=None):
        self.env = env
        self.episodes = episodes
        self.label = label

    def evaluate(self, agent):
        agent.policy.eval()

        wins = []
        rewards = []

        with torch.no_grad():
            for i in range(self.episodes):
                obs, _ = self.env.reset(seed=agent.seed + i)
                done = False
                ep_reward = 0.0

                while not done:
                    action = agent.get_action(
                        obs, noise=False, eval_mode=True
                    )
                    obs, reward, done, trunc, info = self.env.step(action)
                    done = done or trunc
                    ep_reward += reward

                wins.append(1 if info.get("winner", 0) == 1 else 0)
                rewards.append(ep_reward)

        agent.policy.train()

        return float(np.mean(wins)), float(np.mean(rewards))

