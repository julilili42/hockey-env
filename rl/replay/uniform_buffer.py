import numpy as np
import torch
from rl.replay.base_buffer import BaseReplayBuffer


class UniformReplayBuffer(BaseReplayBuffer):
    def sample(self, batch_size):

        if self.size < batch_size:
            batch_size = self.size

        inds = (np.random.rand(batch_size) * self.size).astype(int)
        batch = self.buffer[inds]

        state = torch.tensor(
            np.array([b["state"] for b in batch]),
            dtype=torch.float32,
        )
        action = torch.tensor(
            np.array([b["action"] for b in batch]),
            dtype=torch.float32,
        )
        reward = torch.tensor(
            np.array([b["reward"] for b in batch]),
            dtype=torch.float32,
        )
        next_state = torch.tensor(
            np.array([b["next_state"] for b in batch]),
            dtype=torch.float32,
        )
        done = torch.tensor(
            np.array([b["done"] for b in batch], dtype=np.float32)
        )

        return state, action, reward, next_state, done
