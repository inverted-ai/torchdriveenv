import os
import gymnasium as gym

from torchdriveenv.gym_env import SingleAgentWrapper, WaypointSuiteEnv

__version__ = "0.1.0"

_env_config_path = [os.path.join(x, 'env_configs') for x in __path__]
_data_path = [os.path.join(x, 'data') for x in __path__]

gym.register('torchdriveenv-v0', entry_point=lambda args: SingleAgentWrapper(WaypointSuiteEnv(cfg=args['cfg'], data=args['data'])))
