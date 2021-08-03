# rlext
Library-agnostic helpers and gym extensions for reinforcement learning experiments in Python.
I aim to build this module to collect some independent, tiny and easy-to-use components for my research.
This library only depends on `gym` and some standard scientific Python libraries
(e.g., `numpy` and `pandas`) for easy installation.

## CAUTION
Though I'm using this module for my own research, it is still highly work in progress.
This project is open-sourced just for sharing codes to some of my friends.

## Current components
- `environments`
  - `CartPoleSwingUp-v0`-`CartPoleSwingUp-v5`
  - `CartPoleSwingUpContinuous-v0`-`CartPoleSwingUpContinuous-v4`
  - `CartPoleContinuous-v0`, `CartPoleContinuous-v1`
  - `PuddleWorld-v0`
  - `ContinuousPuddleWorld-v0`
- `record`
  - `Record` class for simple logging for stdout and file.
  - By default, it saves a log as `.jsonl` file. You can easily customize this behavior by passing a callback with `t.Callable[[pandas.DataFrame, pathlib.Path], None]`.
- `video`
  - `VideoWriter` class for writing `.avi/.mp4` videos from `gym.render(mode="rgb_array")`

