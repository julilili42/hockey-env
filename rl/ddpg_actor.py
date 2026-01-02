import torch
import numpy as np
from gymnasium import spaces
from replay_buffer import PrioritizedReplayBuffer
from actor_critic import ActorCritic
from scaler import Scaler
from noise import OUActionNoise
from device import device

torch.set_num_threads(1)

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible

    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)


class DDPGAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, env, **userconfig):

        if not isinstance(env.observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(env.observation_space, self))
        if not isinstance(env.action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(env.action_space, self))

        self.env = env
        self.observation_space = env.observation_space
        self.obs_dim=self.observation_space.shape[0]
        self.action_space = env.action_space
        self.action_n = self.action_space.shape[0]
        self.config = {
            "discount": 0.99,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate_actor": 1e-4,
            "learning_rate_critic": 1e-3,
            "hidden_sizes_actor": [128,128],
            "hidden_sizes_critic": [128,128,64],
            "polyak": 0.995,
            "policy_noise": 0.2,       
            "noise_clip": 0.5,         
            "policy_delay": 2,         

        }
        self.config.update(userconfig)
        
        self.scaler = Scaler(env)

        self.ac = ActorCritic(
            obs_dim=self.obs_dim,
            action_space=self.action_space,
            config=self.config, 
            scaler=self.scaler
        )
        
        self.buffer = PrioritizedReplayBuffer(
            max_size=self.config["buffer_size"],
            alpha=0.6
        )
        self.beta = 0.4
        self.beta_inc = 1e-4


        self.ou_noise = OUActionNoise(
            mean=np.zeros(self.action_n),
            std_deviation=0.2 * np.ones(self.action_n)
        )


        self.train_iter = 0
        self.total_updates = 0


    def act(self, observation, noise=True, return_norm=False):
        obs_norm = self.scaler.normalize_obs(observation)      
        act_norm_t = self.ac.act(obs_norm)                     

        if noise:
            n = torch.tensor(self.ou_noise(), dtype=torch.float32, device=device)
            act_norm_t = torch.clamp(act_norm_t + n, -1.0, 1.0)

        if return_norm:
            return act_norm_t.detach().cpu().numpy()

        a_env_t = self.scaler.scale_action(act_norm_t)         
        return a_env_t.detach().cpu().numpy()







    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return self.ac.state()

    def restore_state(self, state):
        self.ac.restore_state(state)

    def reset(self):
        self.ou_noise.reset()

    def get_buffer_values(self):
        to_torch = lambda x: torch.from_numpy(
            np.asarray(x, dtype=np.float32)
        ).to(device)

        data, indices, weights = self.buffer.sample(
            batch_size=self.config['batch_size'],
            beta=self.beta
        )
        self.beta = min(1.0, self.beta + self.beta_inc)
        data = np.array(data, dtype=object)


        s_env     = to_torch(np.stack(data[:,0]))
        a_env     = to_torch(np.stack(data[:,1]))
        reward    = to_torch(np.stack(data[:,2])[:,None])
        s_next_env= to_torch(np.stack(data[:,3]))
        done      = to_torch(np.stack(data[:,4])[:,None])

        weights = weights.to(s_env.device)
        # policy-space (norm)
        s_norm      = self.scaler.normalize_obs(s_env)
        s_next_norm = self.scaler.normalize_obs(s_next_env)
        
        return (s_env, a_env, s_next_env, reward, done, s_norm, s_next_norm, indices, weights)


    def train(self, iter_fit=32):
        if self.buffer.size < 1000:
            return []

        losses = []
        self.train_iter += 1

        for _ in range(iter_fit):
            

            self.total_updates += 1

            s_env, a_env, s_next_env, reward, done, s_norm, s_next_norm, indices, weights = self.get_buffer_values()
            
            loss1, loss2, td_error = self.ac.update_critic(
                s_env, a_env, reward, s_next_env, done, s_next_norm, weights
            )

            priorities = np.clip(td_error + 1e-6, 1e-6, 10.0)
            self.buffer.update_priorities(indices, priorities)

            if self.total_updates % self.config["policy_delay"] == 0:
                actor_loss = self.ac.update_actor(s_norm, s_env)
                self.ac.parameter_update_polyak()

            losses.append((loss1, loss2))
        return losses
