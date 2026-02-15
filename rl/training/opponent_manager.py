import numpy as np
import torch
from hockey.hockey_env import BasicOpponent
from rl.training.self_play import SelfPlayManager


class OpponentManager:
    def __init__(self, agent, config, resume_from=None):
        self.agent = agent
        self.cfg = config
        self.current_strong_prob = 0.0
        self.current_weak_prob = 1.0
        self.resume_from = resume_from


        self.strong_bot = BasicOpponent(weak=False)
        self.weak_bot = BasicOpponent(weak=True)

        self.use_self_play = config.use_self_play

        if self.use_self_play:
            self.self_play = SelfPlayManager(
                agent,
                interval=config.self_play_interval,
                pool_size=config.self_play_pool_size,
            )
        else:
            self.self_play = None

        self.self_play_prob = 0.0
        self.reset_stats()


    def update_schedule(self, episode, max_episodes):
        progress = episode / max_episodes

        if self.cfg.training_mode == "single":
            self._update_single(progress)
        else:
            self._update_joint(progress)


    def _update_single(self, progress):
        """
        Desired schedule:
        - if starting from pretrained weak model:
            Phase A: 70/30 strong/weak
            Phase B: 80/10/10 strong/weak/self-play
        - if training from scratch: (optional) go 100% strong to keep it strict
        """
        if self.resume_from is None:
            # scratch baseline
            self._set_bot_probs(strong=1.0, weak=0.0)
            self.self_play_prob = 0.0
            return

        # pretrained-start schedule
        if progress < 0.6:
            # 70/30 strong/weak
            self._set_bot_probs(strong=0.7, weak=0.3)
            self.self_play_prob = 0.0
        else:
            # 80/10/10 strong/weak/self-play
            self._set_bot_probs(strong=0.8, weak=0.1)
            self.self_play_prob = 0.1


    def _update_joint(self, progress):
        if progress < 0.1:
            strong_prob, weak_prob = 0.0, 1.0
        elif progress < 0.3:
            strong_prob, weak_prob = 0.4, 0.6
        elif progress < 0.6:
            strong_prob, weak_prob = 0.7, 0.3
        else:
            strong_prob, weak_prob = 0.85, 0.15

        self._set_bot_probs(strong_prob, weak_prob)

        if not self.use_self_play:
            self.self_play_prob = 0.0
            return

        if progress < 0.4:
            self.self_play_prob = 0.0
        elif progress < 0.7:
            ramp = (progress - 0.4) / 0.3
            self.self_play_prob = ramp * self.cfg.self_play_max_prob
        else:
            self.self_play_prob = self.cfg.self_play_max_prob


    def _set_bot_probs(self, strong, weak):
        if strong + weak <= 0:
            raise ValueError("Bot probabilities must sum to > 0")

        self.current_strong_prob = strong
        self.current_weak_prob = weak


    def step(self):
        if self.self_play is not None:
            self.self_play.step()


    def select_action(self, obs2):
        r = np.random.rand()
        from hockey.hockey_env import PolicyOpponent
        # Self-play branch
        if (
            self.use_self_play
            and self.self_play is not None
            and self.self_play.get_opponent() is not None
            and r < self.self_play_prob
        ):
            self.stats["self_play"] += 1
            opponent_policy = self.self_play.get_opponent()
            opp = PolicyOpponent(opponent_policy, device=self.agent.device)
            return opp.act(obs2)

        # Bot branch
        strong_p = self.current_strong_prob
        weak_p = self.current_weak_prob


        if strong_p + weak_p <= 0:
            raise ValueError("Bot probabilities must sum to > 0")

        r_bot = np.random.rand()

        if r_bot < strong_p:
            self.stats["strong"] += 1
            return self.strong_bot.act(obs2)
        else:
            self.stats["weak"] += 1
            return self.weak_bot.act(obs2)


    def reset_stats(self):
        self.stats = {
            "strong": 0,
            "weak": 0,
            "self_play": 0,
        }
