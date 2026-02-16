from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym

import hockey.hockey_env

from comprl.client import Agent, launch_client

from rl.td3.agent import TD3Agent
from rl.td3.config import TD3Config


MODEL_PATH = "runs/20260216_113850_single_dual_eval_abcdefg_3(1)/models/td3_best.pt"


class TD3CompetitionAgent(Agent):

    def __init__(self):
        super().__init__()

        self.env = gym.make("Hockey-One-v0", weak_opponent=False)

        config = TD3Config()

        self.td3 = TD3Agent(
            env=self.env,
            config=config,
            h=256,
        )

        checkpoint = torch.load(MODEL_PATH, map_location=self.td3.device)
        self.td3.policy.load_state_dict(checkpoint["policy"])
        self.td3.policy.eval()

    def get_step(self, observation):
        action = self.td3.get_action(
            np.array(observation),
            noise=False,
            eval_mode=True
        )
        return action.tolist()

    def on_start_game(self, game_id) -> None:
        print("Game started")


    def on_end_game(self, result, stats) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} | "
            f"My score: {stats[0]} | Opponent: {stats[1]}"
        )


def initialize_agent(agent_args=None):
    return TD3CompetitionAgent()


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
