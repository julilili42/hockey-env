import torch
import numpy as np
from gymnasium import spaces
import replay_buffer as mem
from actor_critic import ActorCritic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    def __init__(self, observation_space, action_space, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

        self.observation_space = observation_space
        self.obs_dim=self.observation_space.shape[0]
        self.action_space = action_space
        self.action_n = action_space.shape[0]
        self.config = {
            "eps": 0.05,            # Epsilon: noise strength to add to policy
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate_actor": 1e-4,
            "learning_rate_critic": 1e-3,
            "hidden_sizes_actor": [128,128],
            "hidden_sizes_critic": [128,128,64],
            "polyak": 0.995,
            "policy_noise": 0.2,       
            "noise_clip": 0.5,         
            "policy_delay": 1,         

        }
        self.config.update(userconfig)

        self.ac = ActorCritic(
            obs_dim=self.obs_dim,
            action_space=action_space,
            config=self.config
        )
        
        self.buffer = mem.Memory(max_size=self.config["buffer_size"])

        self.train_iter = 0
        self.total_updates = 0


    def act(self, observation, eps=None):
        return self.ac.act(observation, eps)

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return self.ac.state()

    def restore_state(self, state):
        self.ac.restore_state(state)

    def reset(self):
        pass
    
    def get_buffer_values(self):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        data=self.buffer.sample(batch=self.config['batch_size'])

        s = to_torch(np.stack(data[:,0])) # s_t
        a = to_torch(np.stack(data[:,1])) # a_t
        reward = to_torch(np.stack(data[:,2])[:,None]) # reward  (batchsize,1)
        s_next = to_torch(np.stack(data[:,3])) # s_t+1
        done = to_torch(np.stack(data[:,4])[:,None]) # done signal  (batchsize,1)
        
        return s, a, reward, s_next, done


    def update_critic(self, s, a, reward, s_next, done):
        return self.ac.update_critic(s, a, reward, s_next, done)
    
    def update_actor(self, s):
        return self.ac.update_actor(s)

    def train(self, iter_fit=32):
        losses = []
        self.train_iter += 1

        for _ in range(iter_fit):
            s, a, reward, s_next, done = self.get_buffer_values()

            loss1, loss2 = self.update_critic(s, a, reward, s_next, done)

            self.total_updates += 1

            if self.total_updates % self.config["policy_delay"] == 0:
                actor_loss = self.update_actor(s)
                self.ac.parameter_update_polyak()

                losses.append((loss1, loss2, actor_loss.item()))
            else:
                losses.append((loss1, loss2, None))

        return losses
