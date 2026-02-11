import torch
import torch.nn as nn
import numpy as np
from replay_buffer import ReplayBufferPrioritized as ReplayBuffer
from noise import OUActionNoise
from scaler import Scaler
from feedforward import FeedforwardNetwork
from core.config import TD3Config
from core.device import device
from utils.torch_utils import to_torch, weighted_smooth_l1_loss
from utils.logger import Logger
from twin_critic import TwinCritic
from copy import deepcopy


class TD3:
    def __init__(
        self,
        env,
        config: TD3Config,
        zeta=0.97, # for setting beta
        h=64,
    ):
        self.logger = Logger.get_logger()
        

        self.env = env
        self.cfg = config
        self.device = device
        self.total_steps = 0
        self.train_iter = 0

        self._init_hyperparams(config=config, zeta=zeta)

        self.replay_buffer = ReplayBuffer(buffer_size=config.buffer_size, prioritized_replay=config.prioritized_replay)

        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.shape[0]

        self.scaler = Scaler(self.env, debug=self.cfg.debug)



        self.critic = TwinCritic(
            n_obs,
            n_act,
            h,
            action_low=self.scaler.action_low,
            action_high=self.scaler.action_high,
        ).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device)

        self.policy = FeedforwardNetwork(n_obs, n_act, h=h).to(self.device)
        self.target_policy = deepcopy(self.policy).to(self.device)

        for net in (self.target_critic, self.target_policy):
            for p in net.parameters():
                p.requires_grad = False


        self._init_optimizers(config)

        self.noise_generator = self._init_noise()


        self.logger.info(
            f"Network sizes | "
            f"policy_params={sum(p.numel() for p in self.policy.parameters())}, "
            f"critic_params={sum(p.numel() for p in self.critic.parameters())}"
        )

        self.logger.info(
            f"TD3 init | obs={n_obs}, act={n_act}, "
            f"gamma={self.gamma}, batch={self.batch_size}, "
            f"policy_freq={self.policy_update_freq}, "
            f"prio_replay={self.prioritized_replay}"
        )





    def _init_noise(self):
        if self.noise_mode == "ornstein-uhlenbeck":
            return OUActionNoise(
                mean=np.zeros(self.env.action_space.shape),
                std_deviation=self.action_noise_scale
                * np.ones(self.env.action_space.shape),
            )
        if self.noise_mode == "gaussian":
            return lambda: np.random.normal(
                0, self.action_noise_scale, self.env.action_space.shape
            )
        raise ValueError(f"Unknown noise mode: {self.noise_mode}")


    def _init_optimizers(self, config):
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_q, eps=1e-6, weight_decay=config.wd_q
        )

        self.pol_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.lr_pol, eps=1e-6, weight_decay=config.wd_pol
        )

        self.q_loss = weighted_smooth_l1_loss

    def _init_hyperparams(self, config, zeta):
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.start_steps = config.start_steps
        self.policy_update_freq = config.policy_update_freq

        self.rho_actor = 1 - config.tau_actor
        self.rho_critic = 1 - config.tau_critic

        self.noise_mode = config.noise_mode
        self.action_noise_scale = config.action_noise_scale
        self.target_action_noise_scale = config.target_action_noise_scale
        self.target_action_noise_clip = config.target_action_noise_clip

        self.prioritized_replay = config.prioritized_replay
        self.prioritized_replay_eps = 1e-6

        self.zeta = zeta
        self.beta = 1.0 - zeta / 2.0


    def reset(self):
        if self.noise_mode == "ornstein-uhlenbeck":
            self.noise_generator.reset()
            self.logger.debug("OU noise reset")

    def update_targets(self):
        self.update_target(self.critic, self.target_critic, rho=self.rho_critic)
        self.update_target(self.policy, self.target_policy, rho=self.rho_actor)


    def get_policy_action(self, state):
        return self.policy(state)

    def get_target_policy_action(self, state):
        target_action = self.target_policy(state)
        # add normal noise
        noise = to_torch(
            torch.normal(0, self.target_action_noise_scale, size=target_action.shape),
            device=self.device,
        )
        clamped_noise = torch.clamp(
            noise, -self.target_action_noise_clip, self.target_action_noise_clip
        )

        noisy_action = target_action + clamped_noise

        if self.train_iter % 200 == 0:        
            self.logger.debug(
                f"Target policy action | "
                f"target_mean={target_action.mean().item():.3f}, "
                f"noise_std={noise.std().item():.3f}, "
                f"clamped_noise_max={clamped_noise.abs().max().item():.3f}, "
                f"noisy_min={noisy_action.min().item():.3f}, "
                f"noisy_max={noisy_action.max().item():.3f}"
            )

        clipped = torch.clamp(
                    noisy_action,
                    self.scaler.action_low,
                    self.scaler.action_high,
                )


        if self.train_iter % 200 == 0:
            self.logger.debug(
                f"Clip | "
                f"clipped_min={clipped.min().item():.3f}, "
                f"clipped_max={clipped.max().item():.3f}"
            )
            
        return clipped





    def update_q(self, state, action, reward, next_state, done):
        self.critic_optimizer.zero_grad(set_to_none=True)

        act_target = self.get_target_policy_action(next_state)

        with torch.no_grad():
            q1_target_next, q2_target_next = self.target_critic(next_state, act_target)

        q_target_next = torch.minimum(q1_target_next, q2_target_next)

        target = reward + self.gamma * (1 - done) * q_target_next
        target = target.detach()

        pred1, pred2 = self.critic(state, action)

        if self.train_iter % 200 == 0:
            self.logger.debug(
                f"Critic input | "
                f"state_min={state.min().item():.3f}, "
                f"state_max={state.max().item():.3f}, "
                f"action_min={action.min().item():.3f}, "
                f"action_max={action.max().item():.3f}"
            )

        if self.prioritized_replay:
            probs = self.replay_buffer.get_last_probs()
            weights = (1 / (probs * self.replay_buffer.size)) ** self.beta
            weights = weights / np.max(weights)
            weights = to_torch(weights)
        else:
            weights = None

        loss1 = self.q_loss(pred1, target, weights=weights)
        loss2 = self.q_loss(pred2, target, weights=weights)

        loss = loss1 + loss2

        if not torch.isfinite(loss):
            self.logger.error("Non-finite critic loss detected!")

        loss.backward()

        total_norm = 0
        for p in self.critic.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        if total_norm > 1000:
            self.logger.warning(f"Large critic gradient norm: {total_norm}")

        self.critic_optimizer.step()

        if self.prioritized_replay:
            td_error = (torch.abs(pred1 - target) + torch.abs(pred2 - target)) / 2
            priorities = torch.clamp(td_error, 1e-6, 1e6).detach().cpu().numpy()
            self.replay_buffer.update_priorities(priorities)

        if self.train_iter % 200 == 0:
            self.logger.debug(
                f"Q update | "
                f"target_mean={target.mean().item():.3f}, "
                f"pred1_mean={pred1.mean().item():.3f}, "
                f"pred2_mean={pred2.mean().item():.3f}"
            )

        return loss.item() / 2



    def update_policy(self, state):
        self.pol_optimizer.zero_grad(set_to_none=True)
        # compute loss
        pol_action = self.get_policy_action(state)

        q_pol, _ = self.critic(state, pol_action)

        loss = -q_pol.mean()
        # backpropagate
        loss.backward()
        self.pol_optimizer.step()

        if self.train_iter % 200 == 0:
            self.logger.debug(
                f"Actor update | "
                f"q_pol_mean={q_pol.mean().item():.3f}, "
                f"action_std={pol_action.std().item():.3f}"
            )


        return loss.item()

    def update_step(self, inds=None):
        self.train_iter += 1

        state, action, reward, next_state, done = self.replay_buffer.sample(
            inds=inds, batch_size=self.batch_size
        )

        state = to_torch(state, device=self.device)
        action = to_torch(action, device=self.device)
        reward = to_torch(reward, device=self.device)
        next_state = to_torch(next_state, device=self.device)
        done = to_torch(done, device=self.device)

        critic_loss = self.update_q(state, action, reward, next_state, done)

        actor_loss = None
        if self.train_iter % self.policy_update_freq == 0:
            actor_loss = self.update_policy(state)

        self.update_targets()

        if self.train_iter % 500 == 0:
            self.logger.info(
                f"Train iter {self.train_iter} | "
                f"critic_loss={critic_loss:.4f} | "
                f"actor_loss={actor_loss}"
            )


        return actor_loss, critic_loss

    def get_action(self, state, noise=True, eval_mode=False):
        if not eval_mode:
            self.total_steps += 1

        if self.total_steps < self.start_steps:
            if self.total_steps % 200 == 0:
                self.logger.debug(
                    f"Random action phase | step={self.total_steps}"
                )
            return self.env.action_space.sample()

        state = to_torch(state, device=self.device)
        action = self.get_policy_action(state)  # in [-1,1]

        if noise and not eval_mode:
            noise_np = self.noise_generator()
            noise = torch.tensor(noise_np, device=self.device, dtype=torch.float32)
            action = torch.clamp(action + noise, -1, 1)

        if self.total_steps % 1000 == 0:
            self.logger.debug(
                f"Action step {self.total_steps} | "
                f"policy_mean={action.mean().item():.3f}, "
                f"policy_min={action.min().item():.3f}, "
                f"policy_max={action.max().item():.3f}"
            )

        action = self.scaler.scale_action(action)
        return action.detach().cpu().numpy()



    @staticmethod
    def update_target(net, target, rho=0.995):
        # get state dicts
        target_state_dict = target.state_dict()
        net_state_dict = net.state_dict()
        # update target state dict
        for key in target_state_dict.keys():
            target_state_dict[key] = (
                rho * target_state_dict[key] + (1 - rho) * net_state_dict[key]
            )
        # load target state dict
        target.load_state_dict(target_state_dict)


    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "target_policy": self.target_policy.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }, path)
