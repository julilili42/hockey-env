import gymnasium as gym
import hockey.hockey_env
from hockey.hockey_env import BasicOpponent
import numpy as np
import os

from rl.td3.agent import TD3Agent
from rl.training.train import TD3Trainer
from rl.td3.config import TD3Config
from rl.utils.evaluator import Evaluator
from rl.utils.logger import Logger
from rl.utils.plotter import MetricsPlotter
from rl.utils.metrics import save_metrics
from rl.experiment.directories import create_cluster_run_dirs
from rl.experiment.scheduler import ExperimentScheduler
from rl.experiment.tracking import (
    set_global_seed,
    create_run_info,
    finalize_run_info,
    save_run_info,
    save_config,
)
from rl.experiment.scheduler import ExperimentScheduler
from rl.experiment.definitions import pretrained_vs_scratch


def setup_run_dirs(run_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = create_cluster_run_dirs(run_name, base_dir)
    return (
        dirs["logs"],
        dirs["models"],
        dirs["metrics"],
        dirs["plots"],
        dirs["config"],
    )

def build_envs_and_config(mode, eval_vs_weak):
    if mode == "single":
        config = TD3Config.single()
        train_env = gym.make("Hockey-One-v0")

        eval_env = gym.make("Hockey-One-v0", weak_opponent=eval_vs_weak)
        evaluators = {
            "single": Evaluator(
                eval_env,
                episodes=config.eval_episodes,
                label="SINGLE"
            )
        }

    elif mode == "joint":
        config = TD3Config.joint()
        train_env = gym.make("Hockey-v0")

        strong_env = gym.make("Hockey-One-v0", weak_opponent=False)
        weak_env   = gym.make("Hockey-One-v0", weak_opponent=True)

        evaluators = {
            "strong": Evaluator(strong_env, episodes=config.eval_episodes, label="STRONG"),
            "weak":   Evaluator(weak_env,   episodes=config.eval_episodes, label="WEAK"),
        }

    else:
        raise ValueError("Unknown mode")

    return config, train_env, evaluators


def train_td3(train_env, evaluators, config, model_dir, metrics_dir, plot_dir, episodes, hidden_size, resume_from=None, seed=42):
    total_steps = episodes * config.max_steps

    agent = TD3Agent(
        env=train_env,
        config=config,
        h=hidden_size,
        max_total_steps=total_steps,
        seed=seed,
    )


    if resume_from is not None:
        agent.load(resume_from)
        
    trainer = TD3Trainer(
        agent=agent,
        train_env=train_env,
        evaluators=evaluators,
        model_dir=model_dir,
        metrics_dir=metrics_dir,
        plot_dir=plot_dir,
        max_episodes=episodes,
    )


    trainer.train()
    return trainer


def run_experiment(mode, eval_vs_weak, episodes, hidden_size=256, resume_from=None, seed = 42, external_config=None):
    set_global_seed(seed)

    run_name = f"{mode}_eval_{'weak' if eval_vs_weak else 'strong'}"
    log_dir, model_dir, metrics_dir, plot_dir, config_dir = setup_run_dirs(run_name)

    logger = Logger.get_logger(os.path.join(log_dir, "run.log"))
    logger.info(f"=== NEW RUN STARTED | opponent={run_name} ===")

    config, train_env, evaluators = build_envs_and_config(mode, eval_vs_weak)

    if external_config is not None:
        config = external_config

    run_info = create_run_info(
        config=config,
        episodes_planned=episodes,
        hidden_size=hidden_size,
        eval_vs_weak=eval_vs_weak,
        resume_from=resume_from,
        seed=seed,
    )

    save_config(config, config_dir)

    trainer = train_td3(
        train_env,
        evaluators,
        config,
        model_dir,
        metrics_dir,
        plot_dir,
        episodes,
        hidden_size,
        resume_from=resume_from,
        seed=seed,              
    )


    run_info = finalize_run_info(run_info, trainer)
    save_run_info(run_info, config_dir)

    save_metrics(trainer.metrics, metrics_dir)
    MetricsPlotter(trainer.metrics).save_all(plot_dir)


def get_pretrained_path(name):
    base = os.path.dirname(__file__)
    return os.path.join(base, "pretrained", name)


if __name__ == "__main__":
    scheduler = ExperimentScheduler()

    for exp in pretrained_vs_scratch():
        scheduler.add(exp)

    scheduler.run_all()
