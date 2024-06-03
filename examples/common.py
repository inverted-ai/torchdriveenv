from dataclasses import dataclass
from enum import Enum
from omegaconf import OmegaConf
from typing import Optional

from torchdriveenv.gym_env import EnvConfig
from torchdriveenv.env_utils import construct_env_config


class BaselineAlgorithm(Enum):
    sac = 'sac'
    ppo = 'ppo'
    a2c = 'a2c'
    td3 = 'td3'

@dataclass
class RlCallbackConfig:
    n_steps: int = 1000
    eval_n_episodes: int = 10
    deterministic: bool = True 
    record: bool = True

@dataclass
class WandbCallbackConfig: 
    verbose: bool = True
    gradient_save_freq: int = 100
    model_save_freq: int = 100

@dataclass
class RlTrainingConfig:
    algorithm: BaselineAlgorithm = None
    parallel_env_num: Optional[int] = 2
    project: str = "stable_baselines3" 
    total_timesteps: int = 5e6 
    record_training_examples: bool = True 
    env: EnvConfig = EnvConfig()
    eval_train_callback: RlCallbackConfig = RlCallbackConfig()
    eval_val_callback: RlCallbackConfig = RlCallbackConfig()
    wandb_callback: WandbCallbackConfig = WandbCallbackConfig()

def load_rl_training_config(yaml_path):
    config_from_yaml = OmegaConf.to_object(OmegaConf.load(yaml_path))
    rl_training_config = RlTrainingConfig(**config_from_yaml)
    rl_training_config.algorithm = BaselineAlgorithm(rl_training_config.algorithm)
    rl_training_config.env = construct_env_config(rl_training_config.env)
    
    return rl_training_config
