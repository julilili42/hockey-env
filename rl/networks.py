import torch
import numpy as np

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fun=torch.nn.Tanh(), output_activation=None):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        self.output_activation = output_activation
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [ activation_fun for l in  self.layers ]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x):
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        if self.output_activation is not None:
            return self.output_activation(self.readout(x))
        else:
            return self.readout(x)

    def predict(self, x):
        with torch.no_grad():
            return (
                self.forward(
                    torch.from_numpy(x.astype(np.float32)).to(next(self.parameters()).device)
                )
                .cpu()
                .numpy()
            )

class QNetwork(MLP):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100],
                 learning_rate = 0.0002):
        super().__init__(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes,
                         output_size=1)
        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, observations, actions, targets, weights=None):
        self.optimizer.zero_grad()
        pred = self.Q_value(observations, actions)  # (B,1)

        if weights is None:
            loss = self.loss(pred, targets)
            td_error = torch.abs(pred - targets).detach()
        else:
            w = weights.view(-1, 1)  # (B,1)
            per_sample = torch.nn.functional.smooth_l1_loss(pred, targets, reduction="none")  # (B,1)
            loss = (w * per_sample).mean()
            td_error = torch.abs(pred - targets).detach()

        loss.backward()
        self.optimizer.step()
        return loss.item(), td_error


    def Q_value(self, observations, actions):
        return self.forward(torch.hstack([observations,actions]))
    