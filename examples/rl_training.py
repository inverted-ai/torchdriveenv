import time
import torch
import wandb
import gymnasium as gym
from typing import Any, Dict
import argparse

from stable_baselines3 import SAC, PPO, A2C, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback

import torchdriveenv
from torchdriveenv.env_utils import load_default_train_data, load_default_validation_data

from common import BaselineAlgorithm, load_rl_training_config

training_data = load_default_train_data()
validation_data = load_default_validation_data()

class EvalNTimestepsCallback(BaseCallback):
    """
    Trigger a callback every ``n_steps`` timesteps

    :param n_steps: Number of timesteps between two trigger.
    :param eval_n_episodes: How many episodes to evaluate each time
    """
    def __init__(self, eval_env, n_steps: int, eval_n_episodes: int, deterministic=False, log_tab="eval"):
        super().__init__()
        self.log_tab=log_tab
        self.n_steps = n_steps
        self.eval_n_episodes = eval_n_episodes
        self.deterministic = deterministic
        self.last_time_trigger = 0
        self.eval_env = eval_env

    def _calc_metrics(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        Called after each step
        :param locals_:
        :param globals_:
        """
        info = locals_["info"]
        if "psi_smoothness" not in info:
            return
        self.psi_smoothness_for_single_episode.append(info["psi_smoothness"])
        self.speed_smoothness_for_single_episode.append(info["speed_smoothness"])
        if (info["offroad"] > 0) or (info["collision"] > 0) or (info["traffic_light_violation"] > 0) \
                                 or (info["is_success"]):
            self.episode_num += 1

            if info["offroad"] > 0:
                self.offroad_num += 1
            if info["collision"] > 0:
                self.collision_num += 1
            if info["traffic_light_violation"] > 0:
                self.traffic_light_violation_num += 1
            if info["is_success"]:
                self.success_num += 1
            self.reached_waypoint_nums.append(info["reached_waypoint_num"])
            if len(self.psi_smoothness_for_single_episode) > 0:
                self.psi_smoothness.append(sum(self.psi_smoothness_for_single_episode) / len(self.psi_smoothness_for_single_episode))
            if len(self.speed_smoothness_for_single_episode) > 0:
                self.speed_smoothness.append(sum(self.speed_smoothness_for_single_episode) / len(self.speed_smoothness_for_single_episode))


    def _evaluate(self) -> bool:
        self.episode_num = 0
        self.offroad_num = 0
        self.collision_num = 0
        self.traffic_light_violation_num = 0
        self.success_num = 0
        self.reached_waypoint_nums = []
        self.psi_smoothness = []
        self.speed_smoothness = []

        mean_episode_reward = 0
        mean_episode_length = 0
        for i in range(self.eval_n_episodes):
            self.psi_smoothness_for_single_episode = []
            self.speed_smoothness_for_single_episode = []
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=1,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                callback=self._calc_metrics,
            )
            mean_episode_reward += sum(episode_rewards) / len(episode_rewards)
            mean_episode_length += sum(episode_lengths) / len(episode_lengths)

        mean_episode_reward /= self.eval_n_episodes
        mean_episode_length /= self.eval_n_episodes

        self.logger.record(f"{self.log_tab}/mean_episode_reward", mean_episode_reward)
        self.logger.record(f"{self.log_tab}/mean_episode_length", mean_episode_length)

        self.logger.record(f"{self.log_tab}/offroad_rate", self.offroad_num / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/collision_rate", self.collision_num / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/traffic_light_violation_rate", self.traffic_light_violation_num / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/success_percentage", self.success_num / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/reached_waypoint_num", sum(self.reached_waypoint_nums) / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/psi_smoothness", sum(self.psi_smoothness) / self.eval_n_episodes)
        self.logger.record(f"{self.log_tab}/speed_smoothness", sum(self.speed_smoothness) / self.eval_n_episodes)


    def _on_training_start(self) -> None:
        self._evaluate()


    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            self._evaluate()
        return True

def make_env_(env_config):
    env = gym.make('torchdriveenv-v0', args={'cfg': env_config, 'data': training_data})
    env = Monitor(env)
    return env

def make_val_env_(env_config):
    env = gym.make('torchdriveenv-v0', args={'cfg': env_config, 'data': validation_data})
    env = Monitor(env, info_keywords=("offroad", "collision", "traffic_light_violation"))
    return env

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(
                    prog='tde_examples',
                    description='execute benchmarks for tde')
    parser.add_argument("--config_file", type=str, default="env_configs/single_agent/sac_training.yml")  
    args = parser.parse_args()
    
    rl_training_config = load_rl_training_config(args.config_file)
    env_config = rl_training_config.env

    make_env = lambda: make_env_(env_config)
    make_val_env = lambda: make_val_env_(env_config)
    
    config = {k:v for (k,v) in vars(rl_training_config).items() if isinstance(v, (float, int, str, list, dict, tuple, bool))}
    config.update( {'env-'+k:v for (k,v) in vars(rl_training_config.env).items() if isinstance(v, (float, int, str, list, dict, tuple, bool))})
    config.update( {'tds-'+k:v for (k,v) in vars(rl_training_config.env.simulator).items() if isinstance(v, (float, int, str, list, dict, tuple, bool))})
     
    experiment_name = f"{rl_training_config.algorithm}_{int(time.time())}"
    wandb.init(
        name=experiment_name,
        project=rl_training_config.project,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    env = SubprocVecEnv([make_env] * rl_training_config.parallel_env_num)
    env = VecFrameStack(env, n_stack=rl_training_config.env.frame_stack, channels_order="first")
    
    if rl_training_config.record_training_examples:
        env = VecVideoRecorder(env, "videos",
            record_video_trigger=lambda x: x % 1000 == 0, video_length=200)  # record videos

    if rl_training_config.algorithm == BaselineAlgorithm.sac:
        model = SAC("CnnPolicy", env, verbose=1, tensorboard_log=f"runs/{experiment_name}",
                    policy_kwargs={'optimizer_class':torch.optim.Adam})

    if rl_training_config.algorithm == BaselineAlgorithm.ppo:
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=f"runs/{experiment_name}",
                    policy_kwargs={'optimizer_class':torch.optim.Adam, 'batch_size': 128})

    if rl_training_config.algorithm == BaselineAlgorithm.a2c:
        model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=f"runs/{experiment_name}",
                    policy_kwargs={'optimizer_class':torch.optim.Adam})

    if rl_training_config.algorithm == BaselineAlgorithm.td3:
        model = TD3("CnnPolicy", env, verbose=1, tensorboard_log=f"runs/{experiment_name}",
                    policy_kwargs={'optimizer_class':torch.optim.Adam}, train_freq=1, gradient_steps=1)
 
    eval_val_env = SubprocVecEnv([make_val_env])
    eval_val_env = VecFrameStack(eval_val_env, n_stack=rl_training_config.env.frame_stack, channels_order="first")
    eval_val_callback = EvalNTimestepsCallback(eval_val_env, n_steps=rl_training_config.eval_val_callback['n_steps'], 
                                                 eval_n_episodes=rl_training_config.eval_val_callback['eval_n_episodes'], 
                                                 deterministic=rl_training_config.eval_val_callback['deterministic'], log_tab="eval_val")
    
    if rl_training_config.eval_val_callback['record']:
        eval_val_env = VecVideoRecorder(eval_val_env, "eval_val_video.0_",
            record_video_trigger=lambda x: x % 1000 == 0, video_length=200)  # record videos

    eval_train_env = SubprocVecEnv([make_env])
    eval_train_env = VecFrameStack(eval_train_env, n_stack=rl_training_config.env.frame_stack, channels_order="first")
    eval_train_callback = EvalNTimestepsCallback(eval_train_env, n_steps=rl_training_config.eval_train_callback['n_steps'], 
                                                 eval_n_episodes=rl_training_config.eval_train_callback['eval_n_episodes'], 
                                                 deterministic=rl_training_config.eval_train_callback['deterministic'], log_tab="eval_train")
    
    if rl_training_config.eval_train_callback['record']:
        eval_train_env = VecVideoRecorder(eval_train_env, "eval_train_video.1_",
            record_video_trigger=lambda x: x % 1000 == 0, video_length=200)  # record videos

    model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[
                      eval_val_callback,
                      eval_train_callback,
                      WandbCallback(
                        verbose=rl_training_config.wandb_callback['verbose'],
                        gradient_save_freq=rl_training_config.wandb_callback['gradient_save_freq'],
                        model_save_freq=rl_training_config.wandb_callback['model_save_freq'],
                        model_save_path=f"models/{experiment_name}",
                    )],
    )
