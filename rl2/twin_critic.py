import torch
import torch.nn as nn
from feedforward import FeedforwardNetwork


class TwinCritic(nn.Module):
    def __init__(self, n_obs, n_act, hidden_size, action_low, action_high):
        super().__init__()

        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self.register_buffer("action_range", action_high - action_low)

        self.q1 = FeedforwardNetwork(
            n_obs + n_act,
            1,
            act_out=nn.Identity(),
            h=hidden_size,
        )

        self.q2 = FeedforwardNetwork(
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
