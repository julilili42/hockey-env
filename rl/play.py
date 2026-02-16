import argparse
import torch
import gymnasium as gym
import hockey.hockey_env

from rl.td3.agent import TD3Agent
from rl.td3.config import TD3Config


MODEL_PATH = "runs/20260216_113850_single_dual_eval_abcdefg_3(1)/models/td3_best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak", action="store_true")
    args = parser.parse_args()

    # Einfach One-v0 benutzen
    env = gym.make(
        "Hockey-One-v0",
        weak_opponent=args.weak
    )

    config = TD3Config()
    agent = TD3Agent(env=env, config=config, h=256)

    checkpoint = torch.load(MODEL_PATH, map_location=agent.device)
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.policy.eval()

    obs, _ = env.reset()

    while True:
        action = agent.get_action(
            obs,
            noise=False,
            eval_mode=True
        )

        obs, reward, done, trunc, info = env.step(action)
        env.render()

        if done or trunc:
            print("Winner:", info.get("winner"))
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
