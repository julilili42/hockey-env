import numpy as np
import torch

class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6):
        self.max_size = max_size
        self.alpha = alpha
        self.pos = 0
        self.size = 0

        self.buffer = [None] * max_size
        self.priorities = np.zeros((max_size,), dtype=np.float32)

    def add_transition(self, transition):
        max_prio = self.priorities.max() if self.size > 0 else 1.0

        self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, beta=0.4):
        batch_size = min(batch_size, self.size)

        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float32)


    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
