import numpy as np
import os
import torch

from rl.utils.logger import Logger
from rl.utils.metrics import MetricsTracker
from rl.utils.model_manager import ModelManager
from rl.utils.evaluator import Evaluator
from rl.utils.metrics import save_metrics
from rl.utils.plotter import MetricsPlotter
from rl.utils.early_stopping import EarlyStopping
from rl.training.opponent_manager import OpponentManager


class TD3Trainer:
    def __init__(
    self,
    agent,
    train_env,
    evaluators,
    model_dir,
    metrics_dir,
    plot_dir,
    max_episodes,
    resume_from=None
):
        self.agent = agent
        self.train_env = train_env
        self.max_episodes = max_episodes
        self.max_steps = agent.cfg.max_steps
        self.train_iters = agent.cfg.train_iters
        self.eval_interval = agent.cfg.eval_interval
        self.resume_from = resume_from

        self.opponent_manager = OpponentManager(
            agent=self.agent,
            config=self.agent.cfg,
            resume_from=self.resume_from
        )

        self.logger = Logger.get_logger()
        self.metrics = MetricsTracker()
        self.evaluators = evaluators
        self.model_manager = ModelManager(model_dir)
        self.early_stopper = EarlyStopping(
            patience=self.agent.cfg.early_patience,
            min_delta=self.agent.cfg.early_min_delta,
            mode="max",
        ) if self.agent.cfg.early_stopping else None


        self.model_dir = model_dir
        self.metrics_dir = metrics_dir
        self.plot_dir = plot_dir


        self.logger.info(
            f"Trainer init | episodes={max_episodes}, "
            f"max_steps={self.max_steps}, "
            f"train_iters={self.train_iters}, "
            f"eval_interval={self.eval_interval}"
        )




    def train(self):
        try:
            for ep in range(1, self.max_episodes + 1):
                if self.opponent_manager is not None:
                    self.opponent_manager.update_schedule(ep, self.max_episodes)

                self._log_episode_start(ep)

                self.current_episode = ep
                ep_reward, steps = self._run_episode()


                self.metrics.log_episode(ep_reward)
                self._log_episode_end(ep, ep_reward, steps)

                actor_loss, critic_loss = self._train_agent(ep)

                self._maybe_evaluate(ep)

                if self.opponent_manager is not None and ep % 200 == 0:
                    stats = self.opponent_manager.stats
                    total = sum(stats.values()) + 1e-8

                    strong_ratio = stats["strong"] / total
                    weak_ratio = stats["weak"] / total
                    sp_ratio = stats["self_play"] / total

                    self.logger.info(
                        f"Opponent dist | "
                        f"strong={strong_ratio:.2f} | "
                        f"weak={weak_ratio:.2f} | "
                        f"self_play={sp_ratio:.2f} | "
                        f"self_play_prob={self.opponent_manager.current_self_play_prob:.2f}"
                    )

                    print(
                        f"[TRAINING MIX] strong={strong_ratio:.2f} "
                        f"weak={weak_ratio:.2f} "
                        f"self_play={sp_ratio:.2f}"
                    )

                    self.opponent_manager.reset_stats()

                    self.metrics.log_opponent_dist(
                        episode=ep,
                        strong=strong_ratio,
                        weak=weak_ratio,
                        self_play=sp_ratio,
                        self_play_prob=self.opponent_manager.current_self_play_prob,
                    )

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted manually.")
            print("Training interrupted.")

        except StopIteration:
            self.logger.info("Training stopped by early stopping.")
            print("Training stopped by early stopping.")

        except Exception as e:
            self.logger.exception(f"Training crashed: {e}")
            print(f"Training crashed: {e}")

        finally:
            self._save_checkpoint()



    def _run_episode(self):
        obs, _ = self.train_env.reset(seed=self.agent.seed + self.current_episode)
        self.agent.reset()

        if self.opponent_manager is not None:
            self.opponent_manager.step()

        ep_reward = 0
        steps = 0

        for _ in range(self.max_steps):
            action1 = self.agent.get_action(obs, noise=True)

            obs2 = self.train_env.unwrapped.obs_agent_two()
            action2 = self.opponent_manager.select_action(obs2)

            joint_action = np.concatenate([action1, action2])
            next_obs, reward, done, trunc, _ = self.train_env.step(joint_action)

            stored_action = action1

            self.agent.replay_buffer.push(
                obs, stored_action, reward, next_obs, done or trunc
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


    def _maybe_evaluate(self, ep):
        if ep % self.eval_interval != 0:
            return

        avg_reward_100 = self.metrics.avg_reward(100)

        if self.resume_from is None:
            wr_weak, r_weak = self.evaluators["weak"].evaluate(self.agent)
            wr_strong, r_strong = 0.0, 0.0

            info = (
                f"[EVAL] ep={ep:5d} | "
                f"WR_weak={wr_weak:.3f} | "
                f"R_weak={r_weak:.2f} | "
                f"R100={avg_reward_100:.2f}"
            )

            score_for_model = wr_weak

        else:
            wr_strong, r_strong = self.evaluators["strong"].evaluate(self.agent)
            wr_weak,   r_weak   = self.evaluators["weak"].evaluate(self.agent)

            info = (
                f"[EVAL] ep={ep:5d} | "
                f"WR_strong={wr_strong:.3f} | "
                f"R_strong={r_strong:.2f} | "
                f"WR_weak={wr_weak:.3f} | "
                f"R_weak={r_weak:.2f} | "
                f"R100={avg_reward_100:.2f}"
            )

            score_for_model = min(wr_strong, wr_weak)


        self.metrics.log_eval(wr_strong, wr_weak, r_strong, r_weak)

        self.logger.info(info)
        print(info)


        if self.early_stopper is not None:
            if self.early_stopper.step(score_for_model):
                raise StopIteration

        self.model_manager.update(
            agent=self.agent,
            score=score_for_model,
            episode=ep,
        )

        save_metrics(self.metrics, self.metrics_dir)
        MetricsPlotter(self.metrics).save_all(self.plot_dir)






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

    

    def _save_checkpoint(self):
        self.logger.info("Saving checkpoint (model + metrics + plots)...")

        save_path = os.path.join(self.model_dir, "td3_last.pt")
        self.agent.save(save_path)

        from rl.utils.metrics import save_metrics
        save_metrics(self.metrics, self.metrics_dir)

        from rl.utils.plotter import MetricsPlotter
        plotter = MetricsPlotter(self.metrics)
        plotter.save_all(self.plot_dir)
