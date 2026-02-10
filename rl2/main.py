import gymnasium as gym
import hockey.hockey_env
from td3 import TD3
from td3train import TD3Trainer   
from core.config import TD3Config
import numpy as np
import os
from utils.logger import Logger



weak_opponent = True

train_env = gym.make("Hockey-One-v0", weak_opponent = weak_opponent)
eval_env = gym.make("Hockey-One-v0", weak_opponent = weak_opponent)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
logger = Logger.get_logger(os.path.join(LOG_DIR, "run.log"))
logger.info("=== NEW RUN STARTED ===")

config = TD3Config()

agent = TD3(
    env=train_env,
    config=config,
    h=256,
)

trainer = TD3Trainer(
    agent=agent,
    train_env=train_env,
    eval_env=eval_env,
    model_dir=MODEL_DIR,
    max_episodes=4_000,
)

trainer.train()

np.savez(
    os.path.join(LOG_DIR, "td3_weak.npz"),
    rewards=trainer.rewards,
    winrate=trainer.winrate,
)


