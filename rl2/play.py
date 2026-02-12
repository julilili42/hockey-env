import torch
import gymnasium as gym
import hockey.hockey_env

from td3_agent import TD3Agent
from core.config import TD3Config

MODEL_PATH = "/Users/julian/Desktop/models/weak/td3_best.pt"   


def main():
    env = gym.make("Hockey-One-v0", weak_opponent=True)

    config = TD3Config()
    agent = TD3Agent(env=env, config=config, h=256)

    checkpoint = torch.load(MODEL_PATH, map_location=agent.device)
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.policy.eval()

    obs, _ = env.reset()

    while True:
        action = agent.get_action(obs, noise=False, eval_mode=True)
        obs, reward, done, trunc, info = env.step(action)

        env.render()

        if done or trunc:
            print("Winner:", info.get("winner"))
            obs, _ = env.reset()

if __name__ == "__main__":
    main()
