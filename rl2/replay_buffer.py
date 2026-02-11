import torch
import numpy as np
from utils.logger import Logger


class ReplayBufferPrioritized:
    def __init__(self, buffer_size, prioritized_replay=True):
        self.logger = Logger.get_logger()

        self.buffer_size = buffer_size
        self.init_weight = 1e8
        self.current_index = 0
        self.size = 0
        self.prioritized_replay = prioritized_replay
        self.last_batch_inds = None

        self.buffer = np.full(buffer_size, None, dtype=object)
        if self.prioritized_replay:
            self.weights = np.full(buffer_size, self.init_weight, dtype=np.float32)

        self.logger.info(
            f"ReplayBuffer init | size={buffer_size}, "
            f"prioritized={self.prioritized_replay}"
        )
        


    def push(self, state, action, reward, next_state, done):
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            self.logger.error("NaN/Inf state pushed to buffer")

        if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
            self.logger.error("NaN/Inf next_state pushed to buffer")

        elem_dict = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }
        self.size = min(self.size + 1, self.buffer_size)
        self.buffer[self.current_index] = elem_dict
        if self.prioritized_replay:
            self.weights[self.current_index] = np.max(self.weights[:self.size]) if self.size > 0 else self.init_weight
        self.current_index = (self.current_index + 1) % self.buffer_size

        if self.size % 5000 == 0:
            self.logger.debug(
                f"ReplayBuffer push | size={self.size}, "
                f"idx={self.current_index}"
            )
        if np.isnan(reward):
            self.logger.error("NaN reward pushed to replay buffer")


    def sample(self, inds=None, batch_size=1):

        if self.size < batch_size:
            self.logger.warning(
                f"Sampling with underfilled buffer: "
                f"size={self.size}, batch={batch_size}"
            )
            batch_size = self.size

        if inds is None:
            inds=self.sample_inds((batch_size, ))

        batch = self.buffer[inds]

        if self.prioritized_replay:
            self.last_batch_inds = inds

        # get state, action, reward, next_state, done from batch as torch tensors
        state = torch.tensor(np.array([elem["state"] for elem in batch]), dtype=torch.float32)
        action = torch.tensor(np.array([elem["action"] for elem in batch]), dtype=torch.float32)
        reward = torch.tensor(np.array([elem["reward"] for elem in batch]), dtype=torch.float32)
        next_state = torch.tensor(
            np.array([elem["next_state"] for elem in batch]), dtype=torch.float32
        )
        done = torch.tensor(np.array([elem["done"] for elem in batch], dtype=np.float32))

        if torch.isnan(state).any():
            self.logger.error("NaN detected in sampled states")

        if torch.isnan(action).any():
            self.logger.error("NaN detected in sampled actions")


        return state, action, reward, next_state, done

    def __len__(self):
        return self.size

    def get_last_probs(self):
        if self.prioritized_replay:
            probs = self.weights[self.last_batch_inds]

            if np.any(np.isnan(probs)) or np.any(probs <= 0):
                self.logger.error(
                    "Invalid sampling probabilities in replay buffer"
                )
            return probs / probs.sum()
        else:
            return None

    def update_priorities(self, priorities):
        if self.prioritized_replay:
            if np.any(np.isnan(priorities)) or np.any(priorities <= 0):
                self.logger.error(
                    f"Invalid priorities: "
                    f"min={priorities.min()}, max={priorities.max()}"
                )
            self.weights[self.last_batch_inds] = priorities
            self.last_batch_inds = None
        else:
            raise ValueError("Replay buffer does not use prioritized replay")

    def sample_inds(self, shape):
        if self.prioritized_replay:
            # sample indices with probability proportional to their weight
            weights = self.weights[: self.size]
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights = np.maximum(weights, 1e-6)
            probs = weights / weights.sum()
            

            if np.prod(shape) > self.size:
                inds = np.random.choice(self.size, size=shape, p=probs, replace=True)
            else:
                inds = np.random.choice(self.size, size=shape, p=probs, replace=True)
            return inds
        else:
            # efficient sampling of random indices
            if isinstance(shape, int):
                shape = (shape,)
            return (np.random.rand(*shape) * self.size).astype(int)
