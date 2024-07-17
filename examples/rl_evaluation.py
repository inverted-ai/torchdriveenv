# general imports 
import time
import torch
import wandb
import gymnasium as gym 
import argparse
import numpy as np
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os

# sb3 imports 
from stable_baselines3 import SAC, PPO, A2C, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import VecEnv, VecMonitor, is_vecenv_wrapped

# tds / tde imports 
import torchdriveenv
from torchdriveenv.env_utils import load_default_train_data, load_default_validation_data
from common import BaselineAlgorithm, load_rl_training_config

def search_folder(rootdir):
    file_list = []
    for root, directories, file in os.walk(rootdir):
        for file in file:
            if(file.endswith(".mp4")):
                file_list.append(root + '/' + file)
    return file_list

def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True, 
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions 
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    # if not isinstance(env, VecEnv):
    #     env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    assert n_envs == 1  

    average_info_dict = {'offroad': [], 'collision': [], 'traffic_light_violation': [], 
                         'reached_waypoint_num': [], 'psi_smoothness': [], 'speed_smoothness': [],
                         'rewards': [], 'episode_lengths': []}
    info_dict = {'offroad': [], 'collision': [], 'traffic_light_violation': [], 
                'reached_waypoint_num': [], 'psi_smoothness': [], 'speed_smoothness': [],
                'rewards': []} 

    #
    observations = env.reset()
    states = None
    done = 0
    episode_length = 0
    episode_counts = 0 
    
    #
    while episode_counts < n_eval_episodes :
         
        # get actions 
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=done,
            deterministic=deterministic,
        ) 
        
        # attempt to step environment
        try:
            new_observations, rewards, dones, infos = env.step(actions) 
            done = dones[0]
            info = infos[0] 
            reward = rewards[0]
            info['rewards'] = reward
            observations = new_observations
            
        # if it breaks we restart
        except: 
            print('warning environment crashed....')
            episode_length = -1
            done = 1
            observations = env.reset()
            info_dict = {'offroad': [], 'collision': [], 'traffic_light_violation': [], 
                'reached_waypoint_num': [], 'psi_smoothness': [], 'speed_smoothness': [],
                'rewards': []} 
        
        # update length of episode
        episode_length += 1
        
        # update running info  
        for k in info_dict.keys(): 
            info[k] = info[k].numpy() if torch.is_tensor(info[k]) else info[k]
            info_dict[k].append(info[k])
        
        # if we have finished add in desired info
        if done and (episode_length>0):
            episode_counts += 1
            average_info_dict['episode_lengths'].append(episode_length) 
            average_info_dict['offroad'].append(info_dict['offroad'][-1])
            average_info_dict['collision'].append(info_dict['collision'][-1])
            average_info_dict['traffic_light_violation'].append(info_dict['traffic_light_violation'][-1])
            average_info_dict['reached_waypoint_num'].append(info_dict['reached_waypoint_num'][-1])
            average_info_dict['speed_smoothness'].append(np.mean(info_dict['speed_smoothness']))
            average_info_dict['rewards'].append(np.sum(info_dict['rewards']))
            average_info_dict['episode_lengths'].append(episode_length)
            
            info_dict = {'offroad': [], 'collision': [], 'traffic_light_violation': [], 
                'reached_waypoint_num': [], 'psi_smoothness': [], 'speed_smoothness': [],
                'rewards': []}  
            episode_length = 0
     
    for k in average_info_dict.keys():
        average_info_dict[k] = np.mean(average_info_dict[k]) 

    return average_info_dict

training_data = load_default_train_data()
validation_data = load_default_validation_data()
 
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
    parser.add_argument("--entity", type=str, default="iai")
    parser.add_argument("--run_project", type=str, default="paper-experiments")
    parser.add_argument("--project", type=str, default="paper-evaluations")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--n_eval_episodes", type=str, default=100)

    args = parser.parse_args() 
    
    rl_training_config = load_rl_training_config(args.config_file)
    env_config = rl_training_config.env

    make_env = lambda: make_env_(env_config)
    make_val_env = lambda: make_val_env_(env_config)
    
    config = {k:v for (k,v) in vars(rl_training_config).items() if isinstance(v, (float, int, str, list, dict, tuple, bool))}
    config.update( {'env-'+k:v for (k,v) in vars(rl_training_config.env).items() if isinstance(v, (float, int, str, list, dict, tuple, bool))})
    config.update( {'tds-'+k:v for (k,v) in vars(rl_training_config.env.simulator).items() if isinstance(v, (float, int, str, list, dict, tuple, bool))})
     
    experiment_name = f"{rl_training_config.algorithm}_{int(time.time())}"
    
    env = SubprocVecEnv([make_env])
    env = VecFrameStack(env, n_stack=rl_training_config.env.frame_stack, channels_order="first")

    api = wandb.Api() 
    run = api.run(args.entity+"/"+args.run_project+"/"+args.run_id)
    run.file("model.zip").download(exist_ok=True)

    experiment_name = f"{rl_training_config.algorithm}_{int(time.time())}"
    run = wandb.init(
        name=experiment_name,
        project=args.project,
        config=config,
        sync_tensorboard=False,
        monitor_gym=True,
        save_code=True,
    )
   
    if rl_training_config.algorithm == BaselineAlgorithm.sac:
        model = SAC.load("model.zip", env, verbose=1)
    elif rl_training_config.algorithm == BaselineAlgorithm.ppo:
        model = PPO.load("model.zip", env, verbose=1) 
    elif rl_training_config.algorithm == BaselineAlgorithm.a2c:
        model = A2C.load("model.zip", env, verbose=1) 
    elif rl_training_config.algorithm == BaselineAlgorithm.td3:
        model = TD3.load("model.zip", env, verbose=1) 
    else:
        raise Exception()
    
    # general print statement to inform what we are evaluating 
    print('evaluating: '+args.entity+"/"+args.project+"/"+args.run_id)

    # generate training data 
    eval_train_env = SubprocVecEnv([make_env])
    eval_train_env = VecFrameStack(eval_train_env, n_stack=rl_training_config.env.frame_stack, channels_order="first") 
    eval_train_env = VecVideoRecorder(eval_train_env, "videos/"+experiment_name+'/training',
            record_video_trigger=lambda x: x % 1000 == 0, video_length=200)  # record videos
    train_eval = evaluate_policy(model, eval_train_env, n_eval_episodes=args.n_eval_episodes, deterministic=True)
    print('Training environment evaluation ...')
    for k in train_eval.keys():
        print(k + ": " + str(train_eval[k]))
    
    # generate validation data
    eval_val_env = SubprocVecEnv([make_val_env])
    eval_val_env = VecFrameStack(eval_val_env, n_stack=rl_training_config.env.frame_stack, channels_order="first") 
    eval_val_env = VecVideoRecorder(eval_val_env, "videos/"+experiment_name+'/validation',
            record_video_trigger=lambda x: x % 1000 == 0, video_length=200)  # record videos
    val_eval = evaluate_policy(model, eval_val_env, n_eval_episodes=args.n_eval_episodes, deterministic=True)
    print('Validation environment evaluation ...')
    for k in val_eval.keys():
        print(k + ": " + str(val_eval[k]))

    # combine infos
    log_info = {}
    log_info.update({'train/'+k:train_eval[k]for k in train_eval.keys()})
    log_info.update({'val/'+k:val_eval[k]for k in val_eval.keys()}) 

     # grab all the videos
    video_files = search_folder('./videos/'+experiment_name)
    for i, vid in enumerate(video_files):
        if 'val' in vid:
            log_info.update({"val_example_"+str(i): wandb.Video(vid)})
        else:
            log_info.update({"train_example_"+str(i): wandb.Video(vid)})

    # log to wandb
    wandb.log(log_info)

