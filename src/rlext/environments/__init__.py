import gym

_EASY_PARAMS = dict(
    height_threshold=0.0,
    theta_dot_threshold=1.5,
    x_reward_threshold=1.5,
)

_SWINGUP_PARAMS = [
    # Same as bsuite
    dict(start_position="bottom", allow_noop=True),
    # Difficult
    dict(start_position="bottom", allow_noop=True, height_threshold=0.9),
    # No movecost
    dict(start_position="bottom", allow_noop=False),
    # Easy
    dict(start_position="bottom", allow_noop=False, **_EASY_PARAMS),
    # Arbitary start
    dict(start_position="arbitary", allow_noop=False),
    # Arbitary and easy
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
    # xsNormal
    dict(start_position="bottom"),
    # Difficult
    dict(start_position="bottom", height_threshold=0.9),
    # Easy
    dict(
        start_position="bottom",
        height_threshold=0.0,
        theta_dot_threshold=1.5,
        x_reward_threshold=1.5,
    ),
    # Arbitary start
    dict(start_position="arbitary"),
    # Arbitary and easy
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
