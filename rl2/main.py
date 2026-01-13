import gymnasium as gym
import hockey.hockey_env

from td3 import TD3
from td3train import TD3Trainer   

env = gym.make("Hockey-One-v0")

agent = TD3(
    env=env,
    env_string="Hockey-One-v0",
    steps_max=2000,
    batch_size=256,
    start_steps=30_000,   
    h=256,
)

trainer = TD3Trainer(
    agent=agent,
    env=env,
    max_episodes=2000,
    max_steps=2000,
    train_iters=32,
    eval_interval=100,
)

trainer.train()
