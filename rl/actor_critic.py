import torch
from networks import MLP, QNetwork
from noise import OUNoise

class ActorCritic:
  def __init__(self, obs_dim, action_space, config):
    self.obs_dim = obs_dim
    self.config = config
    self.eps = self.config['eps']
    self.rho = self.config['polyak']
    self.action_space = action_space
    self.action_dim = action_space.shape[0]
    self.action_noise = OUNoise((self.action_dim))

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
    self.critic = QNetwork(observation_dim=self.obs_dim,
                          action_dim=self.action_dim,
                          hidden_sizes= self.config["hidden_sizes_critic"],
                          learning_rate = self.config["learning_rate_critic"])
    # target Q Network
    self.critic_target = QNetwork(observation_dim=self.obs_dim,
                                  action_dim=self.action_dim,
                                  hidden_sizes= self.config["hidden_sizes_critic"],
                                  learning_rate = 0)

    self.parameter_update()


  def act(self, observation, eps=None):
    if eps is None:
        eps = self.eps
    
    action = self.actor.predict(observation) + eps*self.action_noise()  # action in -1 to 1 (+ noise)

    low, high = self.action_space.low, self.action_space.high

    return low + (action + 1.0) * 0.5 * (high - low)


  def parameter_update(self):
    with torch.no_grad():
      for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
          p_targ.mul_(self.rho)
          p_targ.add_((1 - self.rho) * p.data)

      for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
          p_targ.mul_(self.rho)
          p_targ.add_((1 - self.rho) * p.data)

  def restore_state(self, state):
    self.actor.load_state_dict(state["actor"])
    self.critic.load_state_dict(state["critic"])
    self.parameter_update()

  def state(self):
    return {
        "actor": self.actor.state_dict(),
        "critic": self.critic.state_dict()
    }

  def update_critic(self, s, a, reward, s_next, done):
    # critic update
    if self.config["use_target_net"]:
        q_next = self.critic_target.Q_value(s_next, self.actor_target.forward(s_next))
    else:
        q_next = self.critic.Q_value(s_next, self.actor.forward(s_next))
    # target
    q_next = q_next.detach()
    gamma=self.config['discount']
    target = reward + gamma * (1.0-done) * q_next

    # optimize the Q objective
    fit_loss = self.critic.fit(s, a, target)

    return fit_loss


  def update_actor(self, s):
    self.actor_optim.zero_grad()
    # predicted action by actor
    a = self.actor.forward(s)

    # q value of state and action by critic
    q = self.critic.Q_value(s, a)
    actor_loss = -torch.mean(q)
    actor_loss.backward()
    self.actor_optim.step()

    return actor_loss
  

  def reset(self):
    self.action_noise.reset()