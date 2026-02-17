import copy
import random
import torch
import numpy as np


class SelfPlayManager:
    def __init__(self, agent, interval=100, pool_size=40):
        self.agent = agent
        self.interval = interval
        self.pool_size = pool_size

        self.episode_counter = 0

        self.pool = []
        self.scores = []  
        self.current_opponent = None


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
        self.scores.append(1.0)  

        if len(self.pool) > self.pool_size:
            self.pool.pop(0)
            self.scores.pop(0)

        print(f"[SELF-PLAY] Snapshot added. Pool size={len(self.pool)}")


    def update_difficulty(self, win):
        if self.current_opponent is None:
            return

        idx = self.pool.index(self.current_opponent)

        if win == 0:
            self.scores[idx] *= 1.2  
        else:
            self.scores[idx] *= 0.95  

        self.scores[idx] = float(np.clip(self.scores[idx], 0.1, 10.0))


    def get_opponent(self):
        if not self.pool:
            return None

        weights = np.array(self.scores)
        probs = weights / weights.sum()

        idx = np.random.choice(len(self.pool), p=probs)
        self.current_opponent = self.pool[idx]

        return self.current_opponent
