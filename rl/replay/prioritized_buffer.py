import numpy as np
import torch
from rl.replay.base_buffer import BaseReplayBuffer


class PrioritizedReplayBuffer(BaseReplayBuffer):
    def __init__(self, buffer_size, init_weight=1e8):
        super().__init__(buffer_size)

        self.init_weight = init_weight
        self.weights = np.full(buffer_size, init_weight, dtype=np.float32)
        self.last_batch_inds = None

    def push(self, state, action, reward, next_state, done):
        super().push(state, action, reward, next_state, done)

        max_weight = (
            np.max(self.weights[: self.size])
            if self.size > 0
            else self.init_weight
        )

        self.weights[self.current_index - 1] = max_weight

    def sample(self, batch_size):

        if self.size < batch_size:
            batch_size = self.size

        weights = self.weights[: self.size]
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.maximum(weights, 1e-6)

        probs = weights / weights.sum()
        inds = np.random.choice(self.size, size=batch_size, p=probs)

        self.last_batch_inds = inds
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

    def get_last_probs(self):
        probs = self.weights[self.last_batch_inds]
        total = probs.sum()
        return probs / total if total > 0 else np.ones_like(probs) / len(probs)

    def update_priorities(self, priorities):
        self.weights[self.last_batch_inds] = priorities
        self.last_batch_inds = None
