import copy
import random
import torch

class SelfPlayManager:
    def __init__(self, agent, interval=500, pool_size=5):
        self.interval = interval
        self.pool_size = pool_size
        self.episode_counter = 0
        self.pool = []
        self.current_opponent = None
        self.agent = agent

    def step(self):
        self.episode_counter += 1

        if self.episode_counter % self.interval == 0:
            self._add_snapshot()

    def _add_snapshot(self):
        snapshot = copy.deepcopy(self.agent.policy)
        snapshot.eval()

        for p in snapshot.parameters():
            p.requires_grad = False

        self.pool.append(snapshot)

        if len(self.pool) > self.pool_size:
            self.pool.pop(0)

        self.current_opponent = random.choice(self.pool)

    def get_opponent(self):
        if self.current_opponent is None:
            return None
        return self.current_opponent
