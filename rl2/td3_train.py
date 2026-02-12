import numpy as np
import os
from utils.logger import Logger
from utils.metrics import MetricsTracker
from utils.model_manager import ModelManager
from utils.evaluator import Evaluator



class TD3Trainer:
    def __init__(
        self,
        agent,
        train_env,
        eval_env,
        model_dir,
        max_episodes=2000,
        max_steps=500,
        train_iters=32,
        eval_interval=200,
    ):
        self.logger = Logger.get_logger()
        self.metrics = MetricsTracker()
        self.evaluator = Evaluator(eval_env, episodes=100)


        self.logger.info(
            f"Trainer init | episodes={max_episodes}, "
            f"max_steps={max_steps}, "
            f"train_iters={train_iters}, "
            f"eval_interval={eval_interval}"
        )

        self.agent = agent
        self.train_env = train_env
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.train_iters = train_iters
        self.eval_interval = eval_interval
        self.model_manager = ModelManager(model_dir)



    def train(self):
        for ep in range(1, self.max_episodes + 1):
            self._log_episode_start(ep)

            ep_reward, steps = self._run_episode()

            self.metrics.log_episode(ep_reward)
            self._log_episode_end(ep, ep_reward, steps)

            actor_loss, critic_loss = self._train_agent(ep)

            self._maybe_evaluate(ep, actor_loss, critic_loss)


    def _run_episode(self):
        obs, _ = self.train_env.reset()
        
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            self.logger.error(f"NaN/Inf in reset obs: {obs}")

        self.agent.reset()

        ep_reward = 0
        steps = 0

        for _ in range(self.max_steps):
            action = self.agent.get_action(obs, noise=True)
            next_obs, reward, done, trunc, _ = self.train_env.step(action)

            if np.any(np.isnan(next_obs)) or np.any(np.isinf(next_obs)):
                self.logger.error(f"NaN/Inf in next_obs BEFORE buffer: {next_obs}")

            if not np.isfinite(reward):
                self.logger.error(f"Non-finite reward: {reward}")

            self.agent.replay_buffer.push(
                obs, action, reward, next_obs, done or trunc
            )

            ep_reward += reward
            obs = next_obs
            steps += 1

            if done or trunc:
                break

        return ep_reward, steps


    def _train_agent(self, ep):
        if self.agent.total_steps <= self.agent.cfg.batch_size:
            self.logger.debug(
                f"Skipping training | "
                f"steps={self.agent.total_steps}, "
                f"batch={self.agent.cfg.batch_size}"
            )
            return None, None

        actor_losses = []
        critic_losses = []

        for _ in range(self.train_iters):
            actor_loss, critic_loss = self.agent.update_step()
            critic_losses.append(critic_loss)
            if actor_loss is not None:
                actor_losses.append(actor_loss)

        actor_loss = float(np.mean(actor_losses)) if actor_losses else None
        critic_loss = float(np.mean(critic_losses))

        if ep % 50 == 0:
            self.logger.info(
                f"Train ep {ep} | "
                f"critic_loss={critic_loss:.4f} | "
                f"actor_loss={actor_loss}"
            )

        self.metrics.log_update(actor_loss, critic_loss)

        return actor_loss, critic_loss


    def _maybe_evaluate(self, ep, actor_loss, critic_loss):
        if ep % self.eval_interval != 0:
            return

        wr = self.evaluator.evaluate(self.agent)

        avg_reward_100 = self.metrics.avg_reward(100)

        info = (
            f"Eval ep {ep} | "
            f"winrate={wr:.3f} | "
            f"avg_reward_100={avg_reward_100:.2f} | "
            f"actor_loss={actor_loss} | "
            f"critic_loss={critic_loss}"
        )

        self.logger.info(info)
        print(info)

        self.metrics.log_eval(wr)

        self.model_manager.update(
            agent=self.agent,
            score=wr,
            episode=ep,
        )


    def _log_episode_start(self, ep):
        if ep % 100 == 0:
            self.logger.info(f"Episode {ep} started")

    def _log_episode_end(self, ep, reward, steps):
        if np.isnan(reward):
            self.logger.error(f"NaN episode reward at episode {ep}")

        if ep % 100 == 0:
            self.logger.debug(
                f"Episode {ep} finished | steps={steps}, reward={reward:.2f}"
            )