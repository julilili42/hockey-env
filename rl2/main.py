import gymnasium as gym
import hockey.hockey_env
import numpy as np
import os

from td3_agent import TD3Agent
from td3_train import TD3Trainer
from core.config import TD3Config
from utils.logger import Logger
from utils.plotter import MetricsPlotter
from utils.metrics import save_metrics

def setup_run_dirs(base_dir, run_name):
    log_dir = os.path.join(base_dir, "logs", run_name)
    model_dir = os.path.join(base_dir, "models", run_name)
    metrics_dir = os.path.join(base_dir, "metrics", run_name)
    plot_dir = os.path.join(base_dir, "plots", run_name)

    for d in (log_dir, model_dir, metrics_dir, plot_dir):
        os.makedirs(d, exist_ok=True)

    return log_dir, model_dir, metrics_dir, plot_dir


def train_td3(train_env, eval_env, config, model_dir, episodes, hidden_size):
    agent = TD3Agent(env=train_env, config=config, h=hidden_size)

    trainer = TD3Trainer(
        agent=agent,
        train_env=train_env,
        eval_env=eval_env,
        model_dir=model_dir,
        max_episodes=episodes,
    )

    trainer.train()
    return trainer


def run_experiment(weak_opponent, episodes, hidden_size=256):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_name = "weak" if weak_opponent else "strong"

    log_dir, model_dir, metrics_dir, plot_dir = setup_run_dirs(base_dir, run_name)

    logger = Logger.get_logger(os.path.join(log_dir, "run.log"))
    logger.info(f"=== NEW RUN STARTED | opponent={run_name} ===")

    train_env = gym.make("Hockey-One-v0", weak_opponent=weak_opponent)
    eval_env = gym.make("Hockey-One-v0", weak_opponent=weak_opponent)

    config = TD3Config()

    trainer = train_td3(
        train_env,
        eval_env,
        config,
        model_dir,
        episodes,
        hidden_size,
    )

    save_metrics(trainer.metrics, metrics_dir)

    plotter = MetricsPlotter(trainer.metrics)
    plotter.save_all(plot_dir)


if __name__ == "__main__":
    run_experiment(
        weak_opponent=True,
        episodes=1_000,
    )
