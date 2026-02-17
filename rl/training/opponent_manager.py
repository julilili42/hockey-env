import numpy as np
from hockey.hockey_env import BasicOpponent
from hockey.hockey_env import PolicyOpponent
from rl.training.self_play import SelfPlayManager
from rl.training.curricula import CURRICULA


class OpponentManager:
    def __init__(self, agent, config, resume_from=None):
        self.agent = agent
        self.cfg = config
        self.current_strong_prob = 0.0
        self.current_weak_prob = 1.0
        self.resume_from = resume_from
        self.curriculum = CURRICULA[config.curriculum_name]



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

        self.current_self_play_prob = 0.0
        self.reset_stats()


    def update_schedule(self, episode, max_episodes):
        progress = episode / max_episodes
        self._update_single(progress)


    def _update_single(self, progress):
        for threshold, strong, weak, self_play in self.curriculum:
            if progress < threshold:
                self._set_bot_probs(strong, weak, self_play)
                return


    def _set_bot_probs(self, strong, weak, self_play):
        if strong + weak + self_play <= 0:
            raise ValueError("Bot probabilities must sum to > 0")

        self.current_strong_prob = strong
        self.current_weak_prob = weak
        self.current_self_play_prob = self_play

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
            and r < self.current_self_play_prob
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

    def register_outcome(self, winner):
        if self.self_play is not None:
            self.self_play.update_difficulty(winner)

