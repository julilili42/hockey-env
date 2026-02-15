from dataclasses import dataclass

@dataclass
class TD3Config:
    training_mode: str = "joint"  
    # "joint"  -> Hockey-v0 (8 actions)
    # "single" -> Hockey-One-v0 (4 actions)

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
    beta: float = 0.515

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


    # Exploration scheduling
    use_noise_annealing: bool = False
    noise_anneal_mode: str = "linear"   # "linear" | "exp"
    noise_min_scale: float = 0.05

    # Self-play
    use_self_play: bool = True
    self_play_interval: int = 500
    self_play_pool_size: int = 5
    self_play_max_prob: float = 0.7



    @staticmethod
    def single():
        return TD3Config(
            training_mode="single",
            use_self_play=False,
            lr_q=4e-4,
            lr_pol=4e-4,
            wd_q=0.0,
            wd_pol=0.0,
            buffer_size=100_000,
            prioritized_replay=True,
            beta = 0.4, 
            start_steps=2_000,
            action_noise_scale=0.2,
            target_action_noise_scale=0.2,
            target_action_noise_clip=0.3,
            noise_mode="gaussian",
            early_stopping=False,
            early_patience=15,
            use_noise_annealing=True,
            noise_min_scale = 0.1
        )

    @staticmethod
    def joint():
        return TD3Config(
            training_mode="joint",
            lr_q=3e-4,
            lr_pol=3e-4,
            wd_q=0.0,
            wd_pol=0.0,
            train_iters = 32,
            buffer_size=500_000,
            prioritized_replay=False,
            start_steps=3_000,
            action_noise_scale=0.15,
            noise_mode="gaussian",
            early_stopping=False,
            use_self_play=True,
            self_play_interval = 300,
            self_play_pool_size = 8,
            self_play_max_prob = 0.8,
            target_action_noise_scale = 0.1,
            target_action_noise_clip = 0.2
        )
