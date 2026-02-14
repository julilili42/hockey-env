import torch
import numpy as np
from copy import deepcopy
from rl.common.scaler import Scaler
from rl.common.device import device
from rl.common.noise import OrnsteinUhlenbeckNoise, GaussianNoise, PinkNoise, UniformNoise
from rl.td3.networks import ActorNetwork
from rl.td3.config import TD3Config
from rl.td3.networks import TwinQNetwork
from rl.td3.learner import TD3Learner
from rl.utils.torch_utils import to_torch
from rl.utils.logger import Logger
from rl.replay.uniform_buffer import UniformReplayBuffer
from rl.replay.prioritized_buffer import PrioritizedReplayBuffer


class TD3Agent:
    def __init__(
        self,
        env,
        config: TD3Config, 
        h=64,
        max_total_steps=None,
        seed=None
    ):
        self.logger = Logger.get_logger()
        
        self.seed = seed
        self.env = env
        self.cfg = config
        self.device = device
        self.total_steps = 0

        self.beta = self.cfg.beta
        self.current_noise_scale = self.cfg.action_noise_scale
        self.initial_noise_scale = self.cfg.action_noise_scale
        self.max_total_steps = max_total_steps




        if config.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size=config.buffer_size
            )
        else:
            self.replay_buffer = UniformReplayBuffer(
                buffer_size=config.buffer_size
            )

        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.shape[0]

        self.scaler = Scaler(self.env, debug=self.cfg.debug)
        
        self.policy, self.target_policy, self.critic, self.target_critic = self._build_networks(n_obs, n_act, h)

        self._init_optimizers()

        self.noise_generator = self._init_noise()

        self.learner = TD3Learner(
            actor=self.policy,
            critic=self.critic,
            target_actor=self.target_policy,
            target_critic=self.target_critic,
            critic_optimizer=self.critic_optimizer,
            actor_optimizer=self.pol_optimizer,
            replay_buffer=self.replay_buffer,
            scaler=self.scaler,
            config=config,
            device=self.device,
            beta=self.beta,
        )


        self.logger.info(
            f"Network sizes | "
            f"policy_params={sum(p.numel() for p in self.policy.parameters())}, "
            f"critic_params={sum(p.numel() for p in self.critic.parameters())}"
        )

        self.logger.info(
            f"TD3 init | obs={n_obs}, act={n_act}, "
            f"gamma={self.cfg.gamma}, batch={self.cfg.batch_size}, "
            f"policy_freq={self.cfg.policy_update_freq}, "
            f"prio_replay={self.cfg.prioritized_replay}"
        )


    def _build_networks(self, n_obs, n_act, h):
        critic = TwinQNetwork(
            n_obs,
            n_act,
            h,
            action_low=self.scaler.action_low,
            action_high=self.scaler.action_high,
        ).to(self.device)

        target_critic = deepcopy(critic).to(self.device)

        policy = ActorNetwork(n_obs, n_act, h=h).to(self.device)
        target_policy = deepcopy(policy).to(self.device)

        for net in (target_critic, target_policy):
            for p in net.parameters():
                p.requires_grad = False

        return policy, target_policy, critic, target_critic


    def _init_noise(self):
        action_dim = self.env.action_space.shape[0]

        if self.cfg.noise_mode == "ornstein-uhlenbeck":
            return OrnsteinUhlenbeckNoise(
                mean=np.zeros(action_dim),
                std_deviation=self.cfg.action_noise_scale
                * np.ones(action_dim),
            )

        if self.cfg.noise_mode == "gaussian":
            return GaussianNoise(
                shape=(action_dim,),
                scale=self.cfg.action_noise_scale
            )

        if self.cfg.noise_mode == "pink":
            return PinkNoise(
                shape=(action_dim,),
                scale=self.cfg.action_noise_scale,
                seq_len=self.cfg.max_steps
            )
        
        if self.cfg.noise_mode == "uniform":
            return UniformNoise(
                shape=(action_dim,),
                scale=self.cfg.action_noise_scale
            )


        raise ValueError(f"Unknown noise mode: {self.cfg.noise_mode}")



    def _init_optimizers(self):
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.lr_q, eps=1e-6, weight_decay=self.cfg.wd_q
        )

        self.pol_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.cfg.lr_pol, eps=1e-6, weight_decay=self.cfg.wd_pol
        )



    def reset(self):
        if hasattr(self.noise_generator, "reset"):
            self.noise_generator.reset()


    def get_policy_action(self, state):
        return self.policy(state)

    def update_step(self, inds=None):
        state, action, reward, next_state, done = \
            self.replay_buffer.sample(
                batch_size=self.cfg.batch_size
            )


        state = to_torch(state, device=self.device)
        action = to_torch(action, device=self.device)
        reward = to_torch(reward, device=self.device)
        next_state = to_torch(next_state, device=self.device)
        done = to_torch(done, device=self.device)

        return self.learner.update(
            state, action, reward, next_state, done
        )



    def get_action(self, state, noise=True, eval_mode=False):
        if not eval_mode:
            self.total_steps += 1

        if self._in_random_phase(eval_mode):
            return self.env.action_space.sample()

        state = to_torch(state, device=self.device)
        with torch.no_grad():
            action = self.policy(state)

        if noise and not eval_mode:
            self._update_noise_scale()
            action = self._add_noise(action)


        action = self.scaler.scale_action(action)

        if noise and not eval_mode and self.total_steps % 2000 == 0:
            self.logger.info(
                f"Exploration | "
                f"noise_scale={self.current_noise_scale}"
            )

        return action.detach().cpu().numpy()
    

    def _in_random_phase(self, eval_mode):
        return not eval_mode and self.total_steps < self.cfg.start_steps


    def _add_noise(self, action):
        noise_np = self.noise_generator()

        scaled_noise = noise_np * (
            self.current_noise_scale / self.initial_noise_scale
        )

        noise = torch.tensor(
            scaled_noise,
            device=self.device,
            dtype=torch.float32
        )

        return torch.clamp(action + noise, -1, 1)

    
    def _update_noise_scale(self):
        if not self.cfg.use_noise_annealing:
            self.current_noise_scale = self.initial_noise_scale
            return
        
        if self.max_total_steps is None:
            return

        progress = min(self.total_steps / self.max_total_steps, 1.0)

        if self.cfg.noise_anneal_mode == "linear":
            scale = self.initial_noise_scale * (1 - progress)

        elif self.cfg.noise_anneal_mode == "exp":
            scale = self.initial_noise_scale * (0.1 ** progress)

        else:
            raise ValueError("Unknown anneal mode")

        self.current_noise_scale = max(scale, self.cfg.noise_min_scale)



    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "target_policy": self.target_policy.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }, path)


    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_policy.load_state_dict(checkpoint["target_policy"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])

        self.logger.info(f"Checkpoint loaded from {path}")
