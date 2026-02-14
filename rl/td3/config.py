from dataclasses import dataclass

@dataclass
class TD3Config:
    training_mode: str = "joint"  
    # "joint"  -> Hockey-v0 (8 actions)
    # "single" -> Hockey-One-v0 (4 actions)


    # TD3 core
    gamma: float = 0.99
    tau_actor: float = 0.005
    tau_critic: float = 0.005
    policy_update_freq: int = 2

    # Optimizer
    lr_q: float = 3e-4
    lr_pol: float = 3e-4
    wd_q: float = 0.0
    wd_pol: float = 0.0

    # Replay buffer
    buffer_size: int = 200_000
    batch_size: int = 256
    prioritized_replay: bool = False

    # Exploration
    start_steps: int = 5_000
    action_noise_scale: float = 0.2
    target_action_noise_scale: float = 0.2
    target_action_noise_clip: float = 0.3
    noise_mode: str = "gaussian"

    # Debug
    debug: bool = True

    # Early stopping
    early_stopping: bool = False
    early_patience: int = 20
    early_min_delta: float = 0.01

    # Self-play
    use_self_play: bool = True
    self_play_interval: int = 500
    self_play_pool_size: int = 5
    self_play_max_prob: float = 0.7



    @staticmethod
    def single():
        return TD3Config(
            training_mode="single",
            lr_q=5e-4,
            lr_pol=5e-4,
            wd_q=1e-4,
            wd_pol=1e-4,
            buffer_size=100_000,
            prioritized_replay=True,
            start_steps=1_000,
            action_noise_scale=0.15,
            target_action_noise_scale=0.2,
            target_action_noise_clip=0.3,
            noise_mode="ornstein-uhlenbeck",
            early_stopping=True,
            early_patience=5,
            use_self_play=False,
        )

    @staticmethod
    def joint():
        return TD3Config(
            training_mode="joint",
            lr_q=3e-4,
            lr_pol=3e-4,
            wd_q=0.0,
            wd_pol=0.0,
            buffer_size=200_000,
            prioritized_replay=False,
            start_steps=5_000,
            action_noise_scale=0.2,
            noise_mode="gaussian",
            early_stopping=False,
            use_self_play=True,
        )
