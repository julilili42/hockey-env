import torch
from networks import MLP, QNetwork
import numpy as np

class ActorCritic:
  def __init__(self, obs_dim, action_space, config):
    self.obs_dim = obs_dim
    self.config = config
    self.eps = self.config['eps']
    self.rho = self.config['polyak']
    self.action_space = action_space
    self.action_dim = action_space.shape[0]

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

    self.parameter_update_hard()


  def act(self, observation, eps=None):
    if eps is None:
        eps = self.eps

    action = self.actor.predict(observation)
    noise = np.random.normal(0, eps, size=action.shape)
    action = action + noise

    low, high = self.action_space.low, self.action_space.high
    return np.clip(low + (action + 1.0) * 0.5 * (high - low), low, high)


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

  def update_critic(self, s, a, reward, s_next, done):
    with torch.no_grad():
      a_next = self.actor_target(s_next)

      noise = torch.randn_like(a_next) * self.config["policy_noise"]
      noise = torch.clamp(noise, -self.config["noise_clip"], self.config["noise_clip"])
      a_next = torch.clamp(a_next + noise, -1.0, 1.0)

      q1_next = self.critic1_target.Q_value(s_next, a_next)
      q2_next = self.critic2_target.Q_value(s_next, a_next)
      q_next = torch.min(q1_next, q2_next)

      target = reward + self.config["discount"] * (1.0 - done) * q_next

    loss1 = self.critic1.fit(s, a, target)
    loss2 = self.critic2.fit(s, a, target)

    return loss1, loss2


  def update_actor(self, s):
    self.actor_optim.zero_grad()
    # predicted action by actor
    a = self.actor.forward(s)

    # q value of state and action by critic
    q = self.critic1.Q_value(s, a)
    actor_loss = -torch.mean(q)
    actor_loss.backward()
    self.actor_optim.step()

    return actor_loss
  