import gym

SWINGUP_PARAMS = [
    # Same as bsuite
    dict(start_position="bottom", allow_noop=True),
    # Difficult
    dict(start_position="bottom", allow_noop=True, height_threshold=0.9),
    # No movecost
    dict(start_position="bottom", allow_noop=False),
    # Easy
    dict(
        start_position="bottom",
        allow_noop=False,
        height_threshold=0.0,
        theta_dot_threshold=1.5,
        x_reward_threshold=1.5,
    ),
    # Arbitary start
    dict(start_position="arbitary", allow_noop=False),
]


for i, param in enumerate(SWINGUP_PARAMS):
    gym.envs.register(
        id=f"CartPoleSwingUp-v{i}",
        entry_point="rlext.environments.cartpole:CartPoleSwingUp",
        max_episode_steps=1000,
        kwargs=param,
        reward_threshold=800,
    )


CONTINUOUS_SWINGUP_PARAMS = [
    # Normal
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
]


for i, param in enumerate(CONTINUOUS_SWINGUP_PARAMS):
    gym.envs.register(
        id=f"CartPoleSwingUpContinuous-v{i}",
        entry_point="rlext.environments.cartpole:CartPoleSwingUpContinuous",
        max_episode_steps=1000,
        kwargs=param,
        reward_threshold=800,
    )
