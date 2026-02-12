from dataclasses import dataclass

@dataclass
class TD3Config:
  gamma = 0.99
  tau_actor = 0.005
  tau_critic = 0.005
  policy_update_freq = 2

  lr_q = 1e-3
  lr_pol = 1e-3
  wd_q = 1e-4
  wd_pol = 1e-4

  buffer_size = 100_000
  batch_size = 256
  prioritized_replay = True

  start_steps = 1_000
  action_noise_scale = 0.15
  target_action_noise_scale = 0.2
  target_action_noise_clip = 0.5
  noise_mode = "ornstein-uhlenbeck"

  debug = True

  early_stopping = True
  early_patience = 5
  early_min_delta = 0.01

