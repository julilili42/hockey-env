import torch
from networks import MLP, QNetwork
import numpy as np
from device import device


class ActorCritic:
  def __init__(self, obs_dim, action_space, config, scaler):
    self.obs_dim = obs_dim
    self.config = config
    self.rho = self.config['polyak']
    self.action_space = action_space
    self.action_dim = action_space.shape[0]

    # Scaler
    self.scaler = scaler

    # Actor
    self.actor = MLP(input_size=self.obs_dim,
                    hidden_sizes= self.config["hidden_sizes_actor"],
                    output_size=self.action_dim,
                    activation_fun = torch.nn.ReLU(),
                    output_activation = torch.nn.Tanh())
    
    self.actor_target = MLP(input_size=self.obs_dim,
                            hidden_sizes= self.config["hidden_sizes_actor"],
                            output_size=self.action_dim,
                            activation_fun = torch.nn.ReLU(),
                            output_activation = torch.nn.Tanh())
    
    self.actor.to(device)
    self.actor_optim=torch.optim.Adam(self.actor.parameters(), lr=self.config["learning_rate_actor"], eps=0.000001)


    # Critics
    # Q Network
    self.critic1 = QNetwork(observation_dim=self.obs_dim,
                          action_dim=self.action_dim,
                          hidden_sizes= self.config["hidden_sizes_critic"],
                          learning_rate = self.config["learning_rate_critic"])
    self.critic2 = QNetwork(observation_dim=self.obs_dim,
                          action_dim=self.action_dim,
                          hidden_sizes= self.config["hidden_sizes_critic"],
                          learning_rate = self.config["learning_rate_critic"])
    # target Q Network
    self.critic1_target = QNetwork(observation_dim=self.obs_dim,
                                  action_dim=self.action_dim,
                                  hidden_sizes= self.config["hidden_sizes_critic"],
                                  learning_rate = 0)
    self.critic2_target = QNetwork(observation_dim=self.obs_dim,
                                  action_dim=self.action_dim,
                                  hidden_sizes= self.config["hidden_sizes_critic"],
                                  learning_rate = 0)


    self.actor.to(device)
    self.actor_target.to(device)

    self.critic1.to(device)
    self.critic2.to(device)
    self.critic1_target.to(device)
    self.critic2_target.to(device)

    self.actor_target.eval()
    self.critic1_target.eval()
    self.critic2_target.eval()

    for p in self.actor_target.parameters():
        p.requires_grad = False
    for p in self.critic1_target.parameters():
        p.requires_grad = False
    for p in self.critic2_target.parameters():
        p.requires_grad = False
    
    self.parameter_update_hard()

    


  # observation must be normalized!
  def act(self, obs_norm_t: torch.Tensor):
    with torch.no_grad():
        a = self.actor(obs_norm_t.to(device))
    return torch.clamp(a, -1.0, 1.0)



  def parameter_update_hard(self):
    self.critic1_target.load_state_dict(self.critic1.state_dict())
    self.critic2_target.load_state_dict(self.critic2.state_dict())
    self.actor_target.load_state_dict(self.actor.state_dict())


  def parameter_update_polyak(self):
    with torch.no_grad():
      for p, p_targ in zip(self.critic1.parameters(), self.critic1_target.parameters()):
          p_targ.mul_(self.rho)
          p_targ.add_((1 - self.rho) * p.data)
      
      for p, p_targ in zip(self.critic2.parameters(), self.critic2_target.parameters()):
          p_targ.mul_(self.rho)
          p_targ.add_((1 - self.rho) * p.data)

      for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
          p_targ.mul_(self.rho)
          p_targ.add_((1 - self.rho) * p.data)

  def restore_state(self, state):
    self.actor.load_state_dict(state["actor"])
    self.critic1.load_state_dict(state["critic1"])
    self.critic2.load_state_dict(state["critic2"])
    self.parameter_update_hard()

  def state(self):
    return {
        "actor": self.actor.state_dict(),
        "critic1": self.critic1.state_dict(),
        "critic2": self.critic2.state_dict()
    }

  def update_critic(self, s_env, a_env, reward, s_next_env, done, s_next_norm, weights):
    with torch.no_grad():
      # target policy im norm-space
      a_next_norm = self.actor_target(s_next_norm)

      noise = torch.randn_like(a_next_norm) * self.config["policy_noise"]
      noise = torch.clamp(noise, -self.config["noise_clip"], self.config["noise_clip"])
      a_next_norm = torch.clamp(a_next_norm + noise, -1.0, 1.0)

      # -> env-space für den Critic
      a_next_env = self.scaler.scale_action(a_next_norm)

      q1_next = self.critic1_target.Q_value(s_next_env, a_next_env)
      q2_next = self.critic2_target.Q_value(s_next_env, a_next_env)
      q_next = torch.min(q1_next, q2_next)

      target = reward + self.config["discount"] * (1.0 - done) * q_next

    loss1, td1 = self.critic1.fit(s_env, a_env, target, weights)
    loss2, td2 = self.critic2.fit(s_env, a_env, target, weights)

    td_error = 0.5 * (td1 + td2)
    td_error = td_error.squeeze(1).cpu().numpy()
    return loss1, loss2, td_error




  def update_actor(self, s_norm, s_env):
    self.actor_optim.zero_grad()

    a_norm = self.actor(s_norm)              # norm-space
    a_env  = self.scaler.scale_action(a_norm) # env-space

    for p in self.critic1.parameters():
      p.requires_grad = False

    q = self.critic1.Q_value(s_env, a_env)
    actor_loss = -q.mean()

    for p in self.critic1.parameters():
        p.requires_grad = True


    actor_loss.backward()
    self.actor_optim.step()
    return actor_loss



  