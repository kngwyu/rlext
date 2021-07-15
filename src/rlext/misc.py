import typing as t

import numpy as np
from gym.spaces import Box


def box_action_scaler(action_space: Box) -> t.Callable[[np.ndarray], np.ndarray]:
    shift = action_space.low
    scale = action_space.high - action_space.low
    return lambda action: scale / (1.0 + np.exp(-action)) + shift
