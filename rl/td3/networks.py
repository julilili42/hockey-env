import torch
import torch.nn as nn


# define ffw network
class ActorNetwork(nn.Module):
    def __init__(
        self, input_size, output_size, act=torch.tanh, act_out=torch.tanh, h=256
    ):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, output_size)
        self.act = act
        self.act_out = act_out

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.act_out(self.fc3(x))

    def copy(self):
        new_network = ActorNetwork(
            input_size=self.fc1.in_features,
            output_size=self.fc3.out_features,
            act=self.act,
            act_out=self.act_out,
            h=self.fc1.out_features,
        )
        new_network.load_state_dict(self.state_dict())
        return new_network



class TwinQNetwork(nn.Module):
    def __init__(self, n_obs, n_act, hidden_size, action_low, action_high):
        super().__init__()

        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self.register_buffer("action_range", action_high - action_low)

        self.q1 = ActorNetwork(
            n_obs + n_act,
            1,
            act_out=nn.Identity(),
            h=hidden_size,
        )

        self.q2 = ActorNetwork(
            n_obs + n_act,
            1,
            act_out=nn.Identity(),
            h=hidden_size,
        )

    def _unscale_action(self, action):
        if torch.isinf(self.action_range).any():
            return action  
        return ((action - self.action_low) / self.action_range) * 2 - 1.0


    def forward(self, state, action):
        action = self._unscale_action(action)
        x = torch.cat([state, action], dim=-1)

        q1 = self.q1(x).squeeze(-1)
        q2 = self.q2(x).squeeze(-1)

        return q1, q2
