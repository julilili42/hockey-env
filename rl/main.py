import gymnasium as gym
import hockey.hockey_env
import numpy as np
import os

from rl.td3.agent import TD3Agent
from rl.training.train import TD3Trainer
from rl.td3.config import TD3Config
from rl.utils.logger import Logger
from rl.utils.plotter import MetricsPlotter
from rl.utils.metrics import save_metrics

def setup_run_dirs(run_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(base_dir, "experiments", run_name)

    log_dir = os.path.join(exp_dir, "logs")
    model_dir = os.path.join(exp_dir, "models")
    metrics_dir = os.path.join(exp_dir, "metrics")
    plot_dir = os.path.join(exp_dir, "plots")

    for d in (log_dir, model_dir, metrics_dir, plot_dir):
        os.makedirs(d, exist_ok=True)

    return log_dir, model_dir, metrics_dir, plot_dir


def train_td3(train_env, eval_env, config, model_dir, metrics_dir, plot_dir, episodes, hidden_size, resume_from=None,):
    agent = TD3Agent(env=train_env, config=config, h=hidden_size)

    if resume_from is not None:
        agent.load(resume_from)
        
    trainer = TD3Trainer(
        agent=agent,
        train_env=train_env,
        eval_env=eval_env,
        model_dir=model_dir,
        metrics_dir=metrics_dir,
        plot_dir=plot_dir,
        max_episodes=episodes,
    )


    trainer.train()
    return trainer


def run_experiment(weak_opponent, episodes, hidden_size=256, resume_from=None):
    run_name = "weak" if weak_opponent else "strong"
    log_dir, model_dir, metrics_dir, plot_dir = setup_run_dirs(run_name)

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
        metrics_dir,
        plot_dir,
        episodes,
        hidden_size,
        resume_from=resume_from,
    )

    save_metrics(trainer.metrics, metrics_dir)

    plotter = MetricsPlotter(trainer.metrics)
    plotter.save_all(plot_dir)

def get_pretrained_path(name):
    base = os.path.dirname(__file__)
    return os.path.join(base, "pretrained", name)


if __name__ == "__main__":
    run_experiment(
        weak_opponent=False,
        episodes=1_000,
        hidden_size = 256,
        resume_from=get_pretrained_path("weak/td3_weak_best.pt")
    )
