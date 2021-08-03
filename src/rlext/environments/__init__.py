import numpy as np
import gym

_EASY_PARAMS = dict(
    height_threshold=0.0,
    theta_dot_threshold=1.5,
    x_reward_threshold=1.5,
)

_SWINGUP_PARAMS = [
    # Same as bsuite
    dict(start_position="bottom", allow_noop=True),
    # Bottom/Difficult
    dict(start_position="bottom", allow_noop=True, height_threshold=0.9),
    # Bottom/No movecost
    dict(start_position="bottom", allow_noop=False),
    # Bottom/Easy
    dict(start_position="bottom", allow_noop=False, **_EASY_PARAMS),
    # Arbitary
    dict(start_position="arbitary", allow_noop=False),
    # Arbitary/Easy
    dict(start_position="arbitary", allow_noop=False, **_EASY_PARAMS),
]


for i, param in enumerate(_SWINGUP_PARAMS):
    gym.envs.register(
        id=f"CartPoleSwingUp-v{i}",
        entry_point="rlext.environments.cartpole:CartPoleSwingUp",
        max_episode_steps=1000,
        kwargs=param,
        reward_threshold=800,
    )


_CONTINUOUS_SWINGUP_PARAMS = [
    # Bottom/same as bsuite
    dict(start_position="bottom"),
    # Bottom/Difficult
    dict(start_position="bottom", height_threshold=0.9),
    # Bottom/Easy
    dict(
        start_position="bottom",
        height_threshold=0.0,
        theta_dot_threshold=1.5,
        x_reward_threshold=1.5,
    ),
    # Arbitary
    dict(start_position="arbitary"),
    # Arbitary/Easy
    dict(start_position="arbitary", **_EASY_PARAMS),
]


for i, param in enumerate(_CONTINUOUS_SWINGUP_PARAMS):
    gym.envs.register(
        id=f"CartPoleSwingUpContinuous-v{i}",
        entry_point="rlext.environments.cartpole:CartPoleSwingUpContinuous",
        max_episode_steps=1000,
        kwargs=param,
        reward_threshold=800,
    )

gym.envs.register(
    id="CartPoleContinuous-v0",
    entry_point="rlext.environments.cartpole:CartPoleContinuous",
    max_episode_steps=200,
    reward_threshold=195.0,
)

gym.envs.register(
    id="CartPoleContinuous-v1",
    entry_point="rlext.environments.cartpole:CartPoleContinuous",
    max_episode_steps=500,
    reward_threshold=475.0,
)


gym.envs.register(
    id="PuddleWorld-v0",
    entry_point="rlext.environments.puddleworld:PuddleWorld",
    max_episode_steps=1000,
    reward_threshold=-1.0,
)

gym.envs.register(
    id="ContinuousPuddleWorld-v0",
    entry_point="rlext.environments.puddleworld:ContinuousPuddleWorld",
    max_episode_steps=1000,
    reward_threshold=-1.0,
)

# Super easy, for debugging
gym.envs.register(
    id="ContinuousPuddleWorld-v100",
    entry_point="rlext.environments.puddleworld:ContinuousPuddleWorld",
    max_episode_steps=1000,
    reward_threshold=-1.0,
    kwargs={
        "puddles": np.array([[[0.0, 0.0], [0.1, 0.0]]]),
        "start_positions": np.array([[0.5, 0.5]]),
    },
)
