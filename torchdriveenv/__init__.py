import gymnasium as gym

from torchdriveenv.gym_env import SingleAgentWrapper, WaypointSuiteEnv

__version__ = "0.1.0"

gym.register('torchdriveenv-v0', entry_point=lambda args: SingleAgentWrapper(WaypointSuiteEnv(cfg=args['cfg'], data=args['data'])))
