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

        self._observation_space = observation_space
        self._obs_dim=self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        self._config = {
            "eps": 0.1,            # Epsilon: noise strength to add to policy
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate_actor": 0.00001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [128,128],
            "hidden_sizes_critic": [128,128,64],
            "update_target_every": 100,
            "use_target_net": True
        }
        self._config.update(userconfig)

        self.ac = ActorCritic(
            obs_dim=self._obs_dim,
            action_space=action_space,
            config=self._config
        )
        
        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        self.train_iter = 0

    def act(self, observation, eps=None):
        return self.ac.act(observation, eps)

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return self.ac.state()

    def restore_state(self, state):
        self.ac.restore_state(state)

    def reset(self):
        self.ac.reset()
    
    def get_buffer_values(self):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        data=self.buffer.sample(batch=self._config['batch_size'])

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
        self.train_iter+=1
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self.ac.hard_update()
        for _ in range(iter_fit):
            # sample from the replay buffer
            s, a, reward, s_next, done = self.get_buffer_values()

            # critic update
            fit_loss = self.update_critic(s, a, reward, s_next, done)

            # actor update
            actor_loss = self.update_actor(s)

            losses.append((fit_loss, actor_loss.item()))

        return losses

