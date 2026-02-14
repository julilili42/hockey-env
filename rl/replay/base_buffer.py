
import numpy as np
import torch
from rl.utils.logger import Logger


class BaseReplayBuffer:
    def __init__(self, buffer_size):
        self.logger = Logger.get_logger()

        self.buffer_size = buffer_size
        self.current_index = 0
        self.size = 0

        self.buffer = np.full(buffer_size, None, dtype=object)

        self.logger.info(
            f"{self.__class__.__name__} init | size={buffer_size}"
        )

    def push(self, state, action, reward, next_state, done):
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            self.logger.error("NaN/Inf state pushed to buffer")

        if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
            self.logger.error("NaN/Inf next_state pushed to buffer")

        elem = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }

        self.buffer[self.current_index] = elem
        self.size = min(self.size + 1, self.buffer_size)
        self.current_index = (self.current_index + 1) % self.buffer_size

        if np.isnan(reward):
            self.logger.error("NaN reward pushed to replay buffer")

    def sample(self, batch_size):
        raise NotImplementedError

    def __len__(self):
        return self.size
