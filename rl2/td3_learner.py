import torch
import numpy as np
from utils.torch_utils import to_torch, weighted_smooth_l1_loss


class TD3Learner:
    def __init__(
        self,
        actor,
        critic,
        target_actor,
        target_critic,
        critic_optimizer,
        actor_optimizer,
        replay_buffer,
        scaler,
        config,
        device,
        beta,
    ):
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic

        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer

        self.replay_buffer = replay_buffer
        self.scaler = scaler
        self.cfg = config
        self.device = device

        self.gamma = config.gamma
        self.policy_freq = config.policy_update_freq
        self.rho_actor = 1 - config.tau_actor
        self.rho_critic = 1 - config.tau_critic

        self.target_noise_scale = config.target_action_noise_scale
        self.target_noise_clip = config.target_action_noise_clip

        self.prioritized = config.prioritized_replay
        self.beta = beta

        self.q_loss = weighted_smooth_l1_loss
        self.train_iter = 0


    def update(self, state, action, reward, next_state, done):
      self.train_iter += 1

      target = self._compute_target(next_state, reward, done)

      critic_loss = self._update_critic(state, action, target)

      actor_loss = None
      if self.train_iter % self.policy_freq == 0:
          actor_loss = self._update_actor(state)
          self._soft_update(self.actor, self.target_actor, self.rho_actor)
          self._soft_update(self.critic, self.target_critic, self.rho_critic)

      return actor_loss, critic_loss
    

    def _compute_target(self, next_state, reward, done):
      with torch.no_grad():
          target_action = self.target_actor(next_state)

          noise = torch.normal(
              0,
              self.target_noise_scale,
              size=target_action.shape,
              device=self.device,
          )

          noise = torch.clamp(
              noise,
              -self.target_noise_clip,
              self.target_noise_clip,
          )

          target_action = torch.clamp(
              target_action + noise,
              self.scaler.action_low,
              self.scaler.action_high,
          )

          q1_target, q2_target = self.target_critic(
              next_state, target_action
          )

          q_target = torch.minimum(q1_target, q2_target)

          return reward + self.gamma * (1 - done) * q_target


    def _update_critic(self, state, action, target):
      self.critic_optimizer.zero_grad(set_to_none=True)

      q1, q2 = self.critic(state, action)

      if self.prioritized:
          probs = self.replay_buffer.get_last_probs()
          weights = (1 / (probs * self.replay_buffer.size)) ** self.beta
          weights = weights / np.max(weights)
          weights = to_torch(weights, device=self.device)
      else:
          weights = None

      loss1 = self.q_loss(q1, target, weights=weights)
      loss2 = self.q_loss(q2, target, weights=weights)

      critic_loss = loss1 + loss2
      critic_loss.backward()
      self.critic_optimizer.step()

      if self.prioritized:
          td_error = (torch.abs(q1 - target) + torch.abs(q2 - target)) / 2
          priorities = torch.clamp(td_error, 1e-6, 1e6).detach().cpu().numpy()
          self.replay_buffer.update_priorities(priorities)

      return critic_loss.item() / 2


    def _update_actor(self, state):
      self.actor_optimizer.zero_grad(set_to_none=True)

      action = self.actor(state)
      q_val, _ = self.critic(state, action)

      actor_loss = -q_val.mean()
      actor_loss.backward()
      self.actor_optimizer.step()

      return actor_loss.item()


    @staticmethod
    def _soft_update(net, target, rho):
        target_state_dict = target.state_dict()
        net_state_dict = net.state_dict()

        for key in target_state_dict.keys():
            target_state_dict[key] = (
                rho * target_state_dict[key]
                + (1 - rho) * net_state_dict[key]
            )

        target.load_state_dict(target_state_dict)
