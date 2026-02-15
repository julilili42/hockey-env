import os
import json
import datetime
import numpy as np
import random
import torch
from dataclasses import asdict


def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_run_info(
    config,
    episodes_planned,
    hidden_size,
    resume_from,
    seed,
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return {
        "run_settings": {
            "episodes_planned": episodes_planned,
            "hidden_size": hidden_size,
            "seed": seed,
            "timestamp": timestamp,
        },
        "environment": {
            "eval_env": "Hockey-One-v0",
            "eval_opponent": "dual",
        },
        "initialization": {
            "used_pretrained": resume_from is not None,
            "pretrained_path": resume_from,
        },
        "training_features": {
            "self_play_enabled": config.use_self_play,
            "self_play_interval": config.self_play_interval,
            "self_play_pool_size": config.self_play_pool_size,
        },
        "td3_core": {
            "gamma": config.gamma,
            "tau_actor": config.tau_actor,
            "tau_critic": config.tau_critic,
            "batch_size": config.batch_size,
            "buffer_size": config.buffer_size,
            "policy_update_freq": config.policy_update_freq,
            "noise_mode": config.noise_mode,
            "action_noise_scale": config.action_noise_scale,
        },
        "early_stopping": {
            "enabled": config.early_stopping,
            "patience": config.early_patience,
            "min_delta": config.early_min_delta,
        },
        "run_result": {}
    }


def finalize_run_info(run_info, trainer):
    run_info["run_result"]["episodes_completed"] = len(
        trainer.metrics.episode_rewards
    )

    run_info["run_result"]["early_stopped"] = (
        trainer.early_stopper.should_stop
        if trainer.early_stopper is not None
        else False
    )

    best_score = trainer.model_manager.best_score
    run_info["run_result"]["best_winrate"] = (
        None if best_score == float("-inf") else best_score
    )

    return run_info


def save_run_info(run_info, config_dir):
    path = os.path.join(config_dir, "run_info.json")
    with open(path, "w") as f:
        json.dump(run_info, f, indent=4)


def save_config(config, config_dir):
    with open(os.path.join(config_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=4)
