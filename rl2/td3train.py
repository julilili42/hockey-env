import numpy as np
import os
from utils.logger import Logger


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

        self.logger.info(
            f"Trainer init | episodes={max_episodes}, "
            f"max_steps={max_steps}, "
            f"train_iters={train_iters}, "
            f"eval_interval={eval_interval}"
        )

        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.train_iters = train_iters
        self.eval_interval = eval_interval
        self.best_winrate = -np.inf
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.rewards = []
        self.winrate = []


    def train(self):
        for ep in range(1, self.max_episodes + 1):
            self._log_episode_start(ep)

            ep_reward, steps = self._run_episode()

            self.rewards.append(ep_reward)
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
        if self.agent.total_steps <= self.agent.batch_size:
            self.logger.debug(
                f"Skipping training | "
                f"steps={self.agent.total_steps}, "
                f"batch={self.agent.batch_size}"
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

        return actor_loss, critic_loss


    def _maybe_evaluate(self, ep, actor_loss, critic_loss):
        if ep % self.eval_interval != 0:
            return

        wr = self.evaluate(episodes=100)
        avg_reward_100 = np.mean(self.rewards[-100:])

        info = (
            f"Eval ep {ep} | "
            f"winrate={wr:.3f} | "
            f"avg_reward_100={avg_reward_100:.2f} | "
            f"actor_loss={actor_loss} | "
            f"critic_loss={critic_loss}"
        )

        self.logger.info(info)
        print(info)

        self.winrate.append((ep, wr))

        if wr > self.best_winrate + 0.01:
            self.best_winrate = wr
            model_path = os.path.join(self.model_dir, "td3_best.pt")
            self.agent.save(model_path)

            self.logger.info(
                f"New best model saved | ep={ep}, winrate={wr:.3f}"
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


    def evaluate(self, episodes=50):
        wins = []

        for i in range(episodes):
            obs, _ = self.eval_env.reset(seed=i)
            done = False

            while not done:
                action = self.agent.get_action(
                    obs, noise=False, eval_mode=True
                )
                obs, _, done, trunc, info = self.eval_env.step(action)
                done = done or trunc

            wins.append(1 if info.get("winner", 0) == 1 else 0)

        winrate = np.mean(wins)

        if np.isnan(winrate):
            self.logger.error("NaN winrate detected during evaluation")

        self.logger.debug(
            f"Evaluation done | episodes={episodes}, wins={sum(wins)}"
        )

        return winrate
