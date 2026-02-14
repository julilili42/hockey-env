import numpy as np
import torch
from hockey.hockey_env import BasicOpponent
from rl.training.self_play import SelfPlayManager


class OpponentManager:
    def __init__(self, agent, config):
        self.agent = agent
        self.cfg = config
        self.current_strong_prob = 0.0
        self.current_weak_prob = 1.0


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

        # ------------------------
        # BOT CURRICULUM
        # ------------------------

        # Phase 1: very short warmup
        if progress < 0.1:
            self.current_strong_prob = 0.0
            self.current_weak_prob = 1.0

        # Phase 2: introduce strong quickly
        elif progress < 0.3:
            self.current_strong_prob = 0.4
            self.current_weak_prob = 0.6

        # Phase 3: strong dominant
        elif progress < 0.6:
            self.current_strong_prob = 0.7
            self.current_weak_prob = 0.3

        # Phase 4: almost full strong
        else:
            self.current_strong_prob = 0.85
            self.current_weak_prob = 0.15


        # ------------------------
        # SELF-PLAY SCHEDULE
        # ------------------------

        if not self.use_self_play:
            self.self_play_prob = 0.0
            return

        # start self-play earlier
        if progress < 0.4:
            self.self_play_prob = 0.0

        # ramp up faster
        elif progress < 0.7:
            ramp = (progress - 0.4) / 0.3
            self.self_play_prob = ramp * self.cfg.self_play_max_prob

        # full self-play
        else:
            self.self_play_prob = self.cfg.self_play_max_prob






    def step(self):
        if self.self_play is not None:
            self.self_play.step()


    def select_action(self, obs2):
        r = np.random.rand()

        # Self-play branch
        if (
            self.use_self_play
            and self.self_play is not None
            and self.self_play.get_opponent() is not None
            and r < self.self_play_prob
        ):
            self.stats["self_play"] += 1
            opponent_policy = self.self_play.get_opponent()

            with torch.no_grad():
                obs2_t = torch.tensor(
                    obs2,
                    dtype=torch.float32,
                    device=self.agent.device,
                )
                return opponent_policy(obs2_t).cpu().numpy()

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
