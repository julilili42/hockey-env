import torch
import gymnasium as gym
import hockey.hockey_env
import numpy as np

from rl.td3.agent import TD3Agent
from rl.td3.config import TD3Config


MODEL_PATH = "/Users/julian/Documents/Projekte/hockey-env/runs/20260215_002841_single_eval_strong_abcdefg_1/models/td3_best.pt"


def main(mode="single"):
    """
    mode = "single" -> Hockey-One-v0 (4 actions)
    mode = "joint"  -> Hockey-v0 (8 actions)
    """

    if mode == "single":
        env = gym.make("Hockey-One-v0", weak_opponent=True)
        config = TD3Config.single()

    elif mode == "joint":
        env = gym.make("Hockey-v0")
        config = TD3Config.joint()

    else:
        raise ValueError("Mode must be 'single' or 'joint'")

    agent = TD3Agent(env=env, config=config, h=256)

    checkpoint = torch.load(MODEL_PATH, map_location=agent.device)
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.policy.eval()

    obs, _ = env.reset()

    while True:
        if mode == "joint":
            action1 = agent.get_action(obs, noise=False, eval_mode=True)

            obs2 = env.unwrapped.obs_agent_two()
            from hockey.hockey_env import BasicOpponent
            strong_bot = BasicOpponent(weak=False)
            action2 = strong_bot.act(obs2)

            joint_action = np.concatenate([action1, action2])
            obs, reward, done, trunc, info = env.step(joint_action)

        else:
            action = agent.get_action(obs, noise=False, eval_mode=True)
            obs, reward, done, trunc, info = env.step(action)

        env.render()

        if done or trunc:
            print("Winner:", info.get("winner"))
            obs, _ = env.reset()


if __name__ == "__main__":
    main(mode="single")   
