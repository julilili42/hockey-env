import gymnasium as gym
import hockey.hockey_env
import numpy as np
import os

from td3 import TD3
from td3train import TD3Trainer
from core.config import TD3Config
from utils.logger import Logger


def run_experiment(
    weak_opponent,
    episodes,
    hidden_size = 256,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_name = "weak" if weak_opponent else "strong"

    log_dir = os.path.join(base_dir, "logs", run_name)
    model_dir = os.path.join(base_dir, "models", run_name)
    metrics_dir = os.path.join(base_dir, "metrics", run_name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    logger = Logger.get_logger(os.path.join(log_dir, "run.log"))
    logger.info(f"=== NEW RUN STARTED | opponent={run_name} ===")

    train_env = gym.make("Hockey-One-v0", weak_opponent=weak_opponent)
    eval_env = gym.make("Hockey-One-v0", weak_opponent=weak_opponent)

    config = TD3Config()

    agent = TD3(
        env=train_env,
        config=config,
        h=hidden_size,
    )

    trainer = TD3Trainer(
        agent=agent,
        train_env=train_env,
        eval_env=eval_env,
        model_dir=model_dir,
        max_episodes=episodes,
    )

    trainer.train()

    np.savez(
        os.path.join(metrics_dir, "metrics.npz"),
        rewards=trainer.rewards,
        winrate=trainer.winrate,
    )


if __name__ == "__main__":
    run_experiment(
        weak_opponent=True,
        episodes=10_000,
    )
