import typing as t

import numpy as np
from gym import logger, spaces
from gym.envs.classic_control import CartPoleEnv

F32_MAX = np.finfo(np.float32).max

__all__ = ["CartPoleSwingUp", "CartPoleSwingUpContinuous"]


class _SwingUpCommon:
    def _forward(self, force: float) -> t.Tuple[float, float, float, float]:
        x, x_dot, theta, theta_dot = self.state
        costheta, sintheta = np.cos(theta), np.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        return x, x_dot, theta, theta_dot

    def _step(self, force: float) -> t.Tuple[float, bool]:
        x, x_dot, theta, theta_dot = self._forward(force)
        done = bool(x < -self.x_threshold or x > self.x_threshold)

        def _reward() -> float:
            is_upright = np.cos(theta) > self._height_threshold
            is_upright &= np.abs(theta_dot) < self._theta_dot_threshold
            is_upright &= np.abs(x) < self._x_reward_threshold
            return 1.0 if is_upright else 0.0 - self._move_cost

        if not done:
            reward = _reward()
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = _reward()
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' after the episode ending.")
            self.steps_beyond_done += 1
            reward = 0.0

        self.state = x, x_dot, theta, theta_dot
        return reward, done

    def _obs(self) -> np.ndarray:
        x, x_dot, theta, theta_dot = self.state
        obs = np.zeros(5, dtype=np.float32)
        obs[0] = x / self.x_threshold
        obs[1] = x_dot / self.x_threshold
        obs[2] = np.sin(theta)
        obs[3] = np.cos(theta)
        obs[4] = theta_dot
        return obs


class CartPoleSwingUp(CartPoleEnv, _SwingUpCommon):
    START_POSITIONS = ["arbitary", "bottom"]
    ACT_TO_FORCE = [-1.0, 1.0, 0.0]

    def __init__(
        self,
        start_position: str = "arbitary",
        height_threshold: float = 0.5,
        theta_dot_threshold: float = 1.0,
        x_reward_threshold: float = 1.0,
        # This is 2.4 in the original CartPole
        x_threshold: float = 3.0,
        # Aloow 'No operation for action'
        allow_noop: bool = False,
        move_cost: float = 0.1,
    ) -> None:
        super().__init__()
        self.x_threshold = x_threshold
        self.start_position = self.START_POSITIONS.index(start_position)
        self._height_threshold = height_threshold
        self._theta_dot_threshold = theta_dot_threshold
        self._x_reward_threshold = x_reward_threshold
        self._move_cost = move_cost
        if allow_noop:
            self.action_space = spaces.Discrete(3)
        self.allow_noop = allow_noop
        high = np.array([1.0, F32_MAX, 1.0, 1.0, F32_MAX])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def step(self, action: int) -> t.Tuple[np.ndarray, float, bool, dict]:
        """
        action: int
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        force = self.force_mag * self.ACT_TO_FORCE[action]
        reward, done = self._step(force)
        return self._obs(), reward, done, {}

    def reset(self) -> np.ndarray:
        self.state = self.np_random.uniform(-0.05, 0.05, size=(4,))
        if self.start_position == 0:
            self.state[2] = self.np_random.uniform(-np.pi, np.pi)
        else:
            self.state[2] += np.pi
        self.steps_beyond_done = None
        return self._obs()


class CartPoleSwingUpContinuous(CartPoleEnv, _SwingUpCommon):
    START_POSITIONS = ["arbitary", "bottom"]

    def __init__(
        self,
        start_position: str = "arbitary",
        height_threshold: float = 0.5,
        theta_dot_threshold: float = 1.0,
        x_reward_threshold: float = 1.0,
        # This is 2.4 in the original CartPole
        x_threshold: float = 3.0,
        move_cost: float = 0.1,
        max_force: float = 1.0,
        min_force: float = -1.0,
        force_mag: float = 10.0,
    ) -> None:
        super().__init__()
        self.x_threshold = x_threshold
        self.start_position = self.START_POSITIONS.index(start_position)
        self._height_threshold = height_threshold
        self._theta_dot_threshold = theta_dot_threshold
        self._x_reward_threshold = x_reward_threshold
        self._move_cost = move_cost
        obs_high = np.array([1.0, F32_MAX, 1.0, 1.0, F32_MAX])
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(np.array([min_force]), np.array([max_force]))
        self._force_clipper = min_force, max_force
        self.force_mag = 10.0

    def step(self, force: float) -> t.Tuple[np.ndarray, float, bool, dict]:
        force = np.clip(force, *self._force_clipper)
        reward, done = self._step(force * self.force_mag)
        return self._obs(), reward, done, {}

    def reset(self) -> np.ndarray:
        self.state = self.np_random.uniform(-0.05, 0.05, size=(4,))
        if self.start_position == 0:
            self.state[2] = self.np_random.uniform(-np.pi, np.pi)
        else:
            self.state[2] += np.pi
        self.steps_beyond_done = None
        return self._obs()


class CartPoleContinuous(CartPoleEnv, _SwingUpCommon):
    def __init__(
        self,
        height_threshold: float = 0.5,
        theta_dot_threshold: float = 1.0,
        x_reward_threshold: float = 1.0,
        x_threshold: float = 2.4,
        move_cost: float = 0.1,
        max_force: float = 1.0,
        min_force: float = -1.0,
        force_mag: float = 10.0,
    ) -> None:
        super().__init__()
        self.x_threshold = x_threshold
        self._height_threshold = height_threshold
        self._theta_dot_threshold = theta_dot_threshold
        self._x_reward_threshold = x_reward_threshold
        self._move_cost = move_cost
        obs_high = np.array([1.0, F32_MAX, 1.0, 1.0, F32_MAX])
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(np.array([min_force]), np.array([max_force]))
        self._force_clipper = min_force, max_force
        self.force_mag = 10.0

    def _step(self, force: float) -> t.Tuple[float, bool]:
        x, x_dot, theta, theta_dot = self._forward(force)
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        def _reward() -> float:
            is_upright = np.cos(theta) > self._height_threshold
            is_upright &= np.abs(theta_dot) < self._theta_dot_threshold
            is_upright &= np.abs(x) < self._x_reward_threshold
            return 1.0 if is_upright else 0.0 - self._move_cost

        if not done:
            reward = _reward()
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = _reward()
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' after the episode ending.")
            self.steps_beyond_done += 1
            reward = 0.0

        self.state = x, x_dot, theta, theta_dot
        return reward, done

    def step(self, force: float) -> t.Tuple[np.ndarray, float, bool, dict]:
        force = np.clip(force, *self._force_clipper)
        reward, done = self._step(force * self.force_mag)
        return self._obs(), reward, done, {}

    def reset(self) -> np.ndarray:
        self.state = self.np_random.uniform(-0.05, 0.05, size=(4,))
        self.steps_beyond_done = None
        return self._obs()
