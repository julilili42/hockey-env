from dataclasses import dataclass

@dataclass
class TD3Config:
    # Training loop
    max_steps: int = 500
    train_iters: int = 32
    eval_interval: int = 200
    eval_episodes: int = 100

    # TD3 core
    gamma: float = 0.99
    tau_actor: float = 0.005
    tau_critic: float = 0.005
    policy_update_freq: int = 2

    # Optimizer
    lr_q: float = 4e-4
    lr_pol: float = 4e-4
    wd_q: float = 0.0
    wd_pol: float = 0.0

    # Replay buffer
    prioritized_replay: bool = False
    beta: float = 0.15
    buffer_size: int = 300_000
    batch_size: int = 256

    # Exploration
    start_steps: int = 2_000
    action_noise_scale: float = 0.2
    target_action_noise_scale: float = 0.2
    target_action_noise_clip: float = 0.3
    noise_mode: str = "gaussian"

    # Exploration scheduling
    use_noise_annealing: bool = True
    noise_anneal_mode: str = "linear"   # "linear" | "exp"
    noise_min_scale: float = 0.07

    # Early stopping
    early_stopping: bool = False
    early_patience: int = 15
    early_min_delta: float = 0.01

    # Self-play
    use_self_play: bool = True
    self_play_interval: int = 250
    self_play_pool_size: int = 12
