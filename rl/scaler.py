import torch
from device import device

class Scaler:
  def __init__(self, env):
    self.obs_low  = torch.tensor(env.observation_space.low,  dtype=torch.float32, device=device)
    self.obs_high = torch.tensor(env.observation_space.high, dtype=torch.float32, device=device)

    self.act_low  = torch.tensor(env.action_space.low,  dtype=torch.float32, device=device)
    self.act_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)

    self.obs_range = torch.where(
        torch.isfinite(self.obs_high - self.obs_low),
        self.obs_high - self.obs_low,
        torch.ones_like(self.obs_high)
    )
    self.act_range = self.act_high - self.act_low


  def normalize_obs(self, obs):   # env -> [-1,1]
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    obs = 2 * (obs - self.obs_low) / self.obs_range - 1
    return torch.clamp(obs, -1.0, 1.0)

  def unnormalize_obs(self, obs_norm):  # [-1,1] -> env  (optional aber sauber)
    obs_norm = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
    obs = self.obs_low + (obs_norm + 1.0) * 0.5 * self.obs_range
    return obs

  def scale_action(self, act_norm):  # [-1,1] -> env
    act_norm = torch.as_tensor(act_norm, dtype=torch.float32, device=device)
    return self.act_low + (act_norm + 1.0) * 0.5 * self.act_range

  def unscale_action(self, act_env): # env -> [-1,1]
    act_env = torch.as_tensor(act_env, dtype=torch.float32, device=device)
    act_norm = 2.0 * (act_env - self.act_low) / self.act_range - 1.0
    return torch.clamp(act_norm, -1.0, 1.0)


  