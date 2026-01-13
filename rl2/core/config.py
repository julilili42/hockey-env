from dataclasses import dataclass

@dataclass
class TD3Config:
  # --- Algorithm ---
  gamma: float = 0.99
  tau_actor: float = 0.005
  tau_critic: float = 0.005
  policy_update_freq: int = 2

  # --- Optimizer ---
  lr_q: float = 1e-3
  lr_pol: float = 1e-3
  wd_q: float = 1e-4
  wd_pol: float = 1e-4

  # --- Network ---
  hidden_size: int = 256

  # --- Replay ---
  buffer_size: int = 100_000
  batch_size: int = 256
  prioritized_replay: bool = True
  prioritized_eps: float = 1e-6

  # --- Exploration ---
  start_steps: int = 50_000
  action_noise_scale: float = 0.15
  target_action_noise_scale: float = 0.2
  target_action_noise_clip: float = 0.5
  noise_mode: str = "ornstein-uhlenbeck"