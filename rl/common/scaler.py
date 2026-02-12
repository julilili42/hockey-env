import torch
import numpy as np
from rl.utils.logger import Logger
from rl.common.device import device


class Scaler:
    def __init__(self, env, debug=False):
        self.logger = Logger.get_logger()

        self.debug = debug
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space


        self.action_low = torch.tensor(self.action_space.low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(self.action_space.high, dtype=torch.float32, device=device)
        self.action_range = self.action_high - self.action_low
        
        
        self.observation_low = torch.tensor(
            self.observation_space.low, dtype=torch.float32, device=device
        )
        self.observation_high = torch.tensor(
            self.observation_space.high, dtype=torch.float32, device=device
        )
        self.observation_range = self.observation_high - self.observation_low

        a_low = np.asarray(self.action_space.low)
        a_high = np.asarray(self.action_space.high)
        o_low = np.asarray(self.observation_space.low)
        o_high = np.asarray(self.observation_space.high)

        self.action_scaling = not (np.isinf(a_low).any() or np.isinf(a_high).any())
        self.observation_scaling = not (np.isinf(o_low).any() or np.isinf(o_high).any())
        self._step = 0


        self.logger.info(
            f"Scaler init | "
            f"action_scaling={self.action_scaling}, "
            f"obs_scaling={self.observation_scaling}"
        )

        self.logger.debug(
            f"Action space | low={self.action_space.low}, high={self.action_space.high}"
        )

        self.logger.debug(
            f"Observation space | low={self.observation_space.low}, high={self.observation_space.high}"
        )


    def scale_action(self, action):
        self._step += 1

        if self.action_scaling:
            if self.debug and self._step % 1000 == 0:
                self.logger.debug("Action scaling disabled (already env-scaled)")
            return action

        scaled = self.action_low + (action + 1.0) * 0.5 * self.action_range

        if self.debug and self._step % 1000 == 0:
            self.logger.debug(
                f"Scale action | "
                f"in=[{action.min().item():.2f},{action.max().item():.2f}] "
                f"out=[{scaled.min().item():.2f},{scaled.max().item():.2f}]"
            )

        return scaled



    def unscale_action(self, action):
        if self.action_scaling:
            return action

        unscaled = ((action - self.action_low) / self.action_range) * 2 - 1.0

        if self.debug and torch.any(torch.abs(unscaled) > 1.1):
            self.logger.warning(
                f"Unscaled action outside [-1,1]: "
                f"min={unscaled.min().item():.2f}, "
                f"max={unscaled.max().item():.2f}"
            )


        return unscaled


    def scale_state(self, state):
        if self.observation_scaling:
            return state
        return self.observation_low + (state + 1.0) * 0.5 * self.observation_range

    def unscale_state(self, state):
        if self.observation_scaling:
            return state

        unscaled = ((state - self.observation_low) / self.observation_range) * 2 - 1.0

        if torch.any(torch.isnan(unscaled)) or torch.any(torch.isinf(unscaled)):
            self.logger.error(
                f"NaN in unscale_state | "
                f"state_min={state.min().item():.3e}, "
                f"state_max={state.max().item():.3e}, "
                f"obs_low_inf={torch.isinf(self.observation_low).any().item()}, "
                f"obs_range_inf={torch.isinf(self.observation_range).any().item()}"
            )


        return unscaled
