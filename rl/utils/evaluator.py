import numpy as np
import torch

class Evaluator:
    def __init__(self, env, episodes=100):
        self.env = env
        self.episodes = episodes

    def evaluate(self, agent):
        agent.policy.eval()

        wins = []
        
        with torch.no_grad():
          for i in range(self.episodes):
              obs, _ = self.env.reset()
              done = False

              while not done:
                  action = agent.get_action(
                      obs, noise=False, eval_mode=True
                  )
                  obs, _, done, trunc, info = self.env.step(action)
                  done = done or trunc

              wins.append(1 if info.get("winner", 0) == 1 else 0)

        agent.policy.train()
        return float(np.mean(wins))
