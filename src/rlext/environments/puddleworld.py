"""Puddleworld from RLPy."""
import typing as t

import gym
import numpy as np

from gym.utils import seeding


class ContinuousPuddleWorld(gym.Env):
    ACTION_SCALE: float = 0.1
    DEFAULT_PUDDLES: np.ndarray = np.array(
        [[[0.1, 0.75], [0.45, 0.75]], [[0.45, 0.4], [0.45, 0.8]]]
    )
    REWARD_UNIT: float = 0.01
    SCREEN_SIZE: int = 400

    def __init__(
        self,
        noise: float = 0.01,
        puddles: np.ndarray = DEFAULT_PUDDLES,
        start_positions: t.Optional[np.ndarray] = None,
    ) -> None:
        self._noise = noise
        self._state = np.empty(2)
        assert (
            puddles.ndim == 3 and puddles.shape[-1] == 2
        ), f"Invalid shape as puddle: {puddle.shape}"

        self._puddles = puddles
        if start_positions is not None:
            assert (
                start_positions.ndim == 2 and start_positions.shape[1] == 2
            ), f"Invalid shape as start positions: {start_positions.shape}"
        self._start_positions = start_positions

        self.action_space = gym.spaces.Box(
            -np.ones(2, dtype=np.float32) * self.ACTION_SCALE,
            np.ones(2, dtype=np.float32) * self.ACTION_SCALE,
        )
        self.observation_space = gym.spaces.Box(
            np.zeros(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
        )
        self.seed()

        # Some visualization stuffs
        self._reward_map = np.zeros((100, 100))
        for i, x in enumerate(np.linspace(0, 1, 100)):
            for j, y in enumerate(np.linspace(0, 1, 100)):
                self._reward_map[j, i] = self._reward(np.array([x, y]))
        self._domain_fig = None
        self._reward_im = None
        self._state_mark = None

    def reset(self) -> np.ndarray:
        if self._start_positions is None:
            self._state = self.np_random.rand(2)
            while self._is_terminal():
                self._state = self.np_random.rand(2)
        else:
            n_start_positions = len(self._start_positions)
            idx = self.np_random.choice(np.arange(n_start_positions))
            self._state = self._start_positions[idx].copy()
        return self._state.copy()

    def seed(self, seed: t.Optional[int] = None) -> t.List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> t.Tuple[np.ndarray, float, bool, dict]:
        action = np.clip(action, -self.ACTION_SCALE, self.ACTION_SCALE).reshape(2)
        ns = self._state + action + self.np_random.randn() * self._noise
        # make sure we stay inside the [0,1]^2 region
        ns = np.minimum(ns, 1.0)
        ns = np.maximum(ns, 0.0)
        self._state = ns.copy()
        return ns, self._reward(ns), self._is_terminal(), {}

    def render(self, _mode: t.Any) -> None:
        s = self._state
        # Draw the environment
        if self._domain_fig is None:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            from rlext.mpl_ion import plt

            self._domain_fig = plt.figure("Puddleworld")
            ax = self._domain_fig.add_subplot(111)
            self._reward_im = ax.imshow(
                self._reward_map, extent=(0, 1, 0, 1), origin="lower", cmap="YlOrBr"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

            divider = make_axes_locatable(ax)
            cbar_ax = divider.append_axes("right", size="6%", pad=0.1)
            self._domain_fig.colorbar(
                self._reward_im, cax=cbar_ax, orientation="vertical"
            )
            cbar_ax.set_ylabel("Reward")

            self._state_mark = ax.plot(s[0], s[1], "bd", markersize=20)
        else:
            self._state_mark[0].set_data([s[0]], [s[1]])
        self._domain_fig.canvas.draw()
        self._domain_fig.canvas.flush_events()

    def _is_terminal(self, state: t.Optional[np.ndarray] = None) -> bool:
        if state is None:
            state = self._state
        return state.sum() > 0.95 * 2

    def _reward(self, s: np.ndarray) -> float:
        if self._is_terminal(s):
            return self.REWARD_UNIT * 10  # goal state reached
        reward = -self.REWARD_UNIT
        # compute puddle influence
        d = self._puddles[:, 1, :] - self._puddles[:, 0, :]
        denom = (d ** 2).sum(axis=1)
        g = ((s - self._puddles[:, 0, :]) * d).sum(axis=1) / denom
        g = np.minimum(g, 1)
        g = np.maximum(g, 0)
        dists = np.sqrt(((self._puddles[:, 0, :] + g * d - s) ** 2).sum(axis=1))
        dists = dists[dists < 0.1]
        if len(dists) > 0:
            reward -= self.REWARD_UNIT * 10 * (0.1 - dists[dists < 0.1]).max()
        return reward


class PuddleWorld(ContinuousPuddleWorld):
    ACTIONS = 0.05 * np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float)

    def __init__(self, noise: float) -> None:
        super().__init__(noise)

        self.action_space = gym.spaces.Discrete(4)

    def step(self, action: int) -> t.Tuple[np.ndarray, float, bool, dict]:
        return super().step(self.ACTIONS[action])
