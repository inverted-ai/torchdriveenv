import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../torchdrivesim/")

import time
import torch
import wandb
import gymnasium as gym

from stable_baselines3 import SAC, PPO, A2C, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, VecFrameStack, SubprocVecEnv, DummyVecEnv

from torchdriveenv.gym_env import BaselineAlgorithm
from torchdriveenv.env_utils import load_waypoint_suite_data, load_rl_training_config, EvalNTimestepsCallback, load_replay_data

from wandb.integration.sb3 import WandbCallback


rl_training_config = load_rl_training_config("env_configs/rl_training.yml")
env_config = rl_training_config.env

if env_config.train_replay_data_path is not None:
    training_data = load_replay_data(env_config.train_replay_data_path)
else:
    training_data = load_waypoint_suite_data("data/training_cases.yml")

if env_config.val_replay_data_path is not None:
    validation_data = load_replay_data(env_config.val_replay_data_path, add_car_seq=False)
else:
    validation_data = load_waypoint_suite_data("data/validation_cases.yml")


def make_env():
    env = gym.make('torchdriveenv-v0', args={'cfg': env_config, 'data': training_data})
    env = Monitor(env)
    return env


def make_val_env():
    eval_env_config = env_config
    eval_env_config.terminated_at_blame = False
#    eval_env_config.log_blame = True
    eval_env_config.ego_only = False
    env = gym.make('torchdriveenv-v0', args={'cfg': eval_env_config, 'data': validation_data})
    env = Monitor(env, info_keywords=("offroad", "collision", "traffic_light_violation"))
    return env


if __name__=='__main__':
    config = {"policy_type": "CnnPolicy", "total_timesteps": 5000000}
    experiment_name = f"collect_expert_dataset_{int(time.time())}"
    wandb.init(
        name=experiment_name,
        project="EBDM",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    env = SubprocVecEnv([make_env] * rl_training_config.parallel_env_num)
    env = VecFrameStack(env, n_stack=3, channels_order="first")
    env = VecVideoRecorder(env, "videos",
        record_video_trigger=lambda x: x % 1000 == 0, video_length=200)  # record videos

    if rl_training_config.algorithm == BaselineAlgorithm.sac:
        model = SAC(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{experiment_name}",
                    policy_kwargs={'optimizer_class':torch.optim.Adam})
#        model = SAC.load("mid_dist_reward_trained_models/BaselineAlgorithm.sac_without_blame_1715828795/model", env=env)

    if rl_training_config.algorithm == BaselineAlgorithm.ppo:
        model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{experiment_name}",
                    policy_kwargs={'optimizer_class':torch.optim.Adam})

    if rl_training_config.algorithm == BaselineAlgorithm.a2c:
        model = A2C(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{experiment_name}",
                    policy_kwargs={'optimizer_class':torch.optim.Adam})

    if rl_training_config.algorithm == BaselineAlgorithm.td3:
        model = TD3(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{experiment_name}",
                    policy_kwargs={'optimizer_class':torch.optim.Adam}, train_freq=1, gradient_steps=1)

    eval_val_env = SubprocVecEnv([make_val_env])
    eval_val_env = VecFrameStack(eval_val_env, n_stack=3, channels_order="first")

    eval_val_callback = EvalNTimestepsCallback(eval_val_env, n_steps=1000, eval_n_episodes=20, deterministic=False, log_tab="eval_val")
    eval_val_env = VecVideoRecorder(eval_val_env, "eval_val_video.0_",
        record_video_trigger=lambda x: x % 1000 == 0, video_length=200)  # record videos

#    eval_train_env = SubprocVecEnv([make_env])
#    eval_train_env = VecFrameStack(eval_train_env, n_stack=3, channels_order="first")
#
#    eval_train_callback = EvalNTimestepsCallback(eval_train_env, n_steps=25000, eval_n_episodes=10, deterministic=False, log_tab="eval_train")
#    eval_train_env = VecVideoRecorder(eval_train_env, "eval_train_video.1_",
#        record_video_trigger=lambda x: x % 1000 == 0, video_length=200)  # record videos
#                  eval_val_callback,
#                      eval_train_callback,

    model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[
                        eval_val_callback,
                        WandbCallback(
                        verbose=1,
                        gradient_save_freq=100,
                        model_save_freq=10,
                        model_save_path=f"models/{experiment_name}",
                    )],
    )
