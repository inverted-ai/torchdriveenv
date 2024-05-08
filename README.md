# TorchDriveEnv
<img src="https://github.com/inverted-ai/torchdriveenv/assets/16724505/78a8b203-6bef-4796-b08d-b65b4139ddb2.gif" alt="Three Way" width="512"> \
TorchDriveEnv is a lightweight reinforcement learning benchmark for autonomous driving written entirely in python, that provides human like non-playable characters (NPCs). Installation is as easy as running:

```
pip install torchdriveenv[baselines]
```

The benchmark provides NPCs through the state of the art behavioral models provided through the [Inverted AI API](https://www.inverted.ai/home). While emphasis is on ease of use, TorchDriveEnv is fully customizable, with easy to define new scenarios or reward. TorchDriveEnv is built on the lightweight driving simulator [TorchDriveSim](https://github.com/inverted-ai/torchdrivesim/), and therefore comes with various kinematic models, agent types and traffic controls. 


## Scenario Overview
TorchDriveEnv comes bundled with a predefined collection of diverse and challenging training and testing scenarios. Training scenarios were created using the Inverted AI API, while the testing scenarios were hand tailored for interesting challenges. An overview of the testing scenarios is given in the visualization below. Note that while the ego agent has fixed waypoints in these scenarios, its starting position is sampled uniformly between the first two waypoints, and NPCs are initialized dynamically using the Inverted AI API. This results in a large variation in the available scenarios. 

| <img src="https://github.com/inverted-ai/torchdriveenv/assets/16724505/ab82ec1e-fe79-4721-a996-512162032894.png" alt="Three Way" width="204"> | <img src="https://github.com/inverted-ai/torchdriveenv/assets/16724505/ce50a190-065f-4f59-b010-1e503ef78696.png" alt="Chicken" width="204"> | <img src="https://github.com/inverted-ai/torchdriveenv/assets/16724505/0ebddde4-62b0-44ad-bf40-bbb029d04589.png" alt="Parked Car" width="204"> | <img src="https://github.com/inverted-ai/torchdriveenv/assets/16724505/d38b72ff-f90c-4f83-8bb5-454f92168d1d.png" alt="Roundabout" width="204"> | <img src="https://github.com/inverted-ai/torchdriveenv/assets/16724505/1d4b8706-0bb6-4793-b57c-2b35eb020650.png" alt="Traffic Lights" width="204"> |
|------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Three Way | Chicken | Parked Car | Roundabout | Traffic Lights |



## Reward
The reward for TorchDriveEnv is based on waypoints that are provided to the ego agent, as well as a reward for movement and smoothness. Episodes that cause infractions (causing collisions or going offroad) are terminated early, otherwise the episode is terminated after a fixed number of steps. For a full overview of the reward, refer to the paper linked below. 

# Installation

The basic installation of torchdriveenv uses an OpenCV renderer, which is slower but easy to install. PyTorch3D renderer can be faster, but it requires specific versions of CUDA and PyTorch, so it is best installed using Docker.

TorchDriveEnv comes with an example that integrates [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master), which can be used for testing purposes.

## Opencv Rendering

To install the “torchdriveenv” with opencv rendering it suffices to simply:
```
pip install torchdriveenv[baselines]
```
An example can be accessed by first cloning this repository and then:
```
cd examples
python rl_training.py
```

## Pytorch3d Rendering

To install torchdriveenv with Pytorch3d rendering, we recommend an installation through docker. A Dockerfile is provided and can be built using:
```
docker build --target torchdriveenv-first-release -t torchdriveenv-first-release:latest .
```

An example can be accessed by first cloning this repository and then:
```
cd examples
docker compose up rl-training
```
## NPCs
To use the NPC's provided through the [Inverted AI API](https://docs.inverted.ai/en/latest/), provide a [valid API key](https://www.inverted.ai/portal/login) through the environment variable `IAI_API_KEY`.

## Wandb
TorchDriveEnv is integrated with [Weights & Biases](https://wandb.ai) for experiment logging, to make use of it provide an [API key](https://docs.wandb.ai/quickstart) through `WANDB_API_KEY`.

## Running Experiments
The results reported in the paper can be reproduced by cloning this repository and running:

``` 
python rl_training.py --config_file=env_configs/multi_agent/sac_training.yml
python rl_training.py --config_file=env_configs/multi_agent/ppo_training.yml
python rl_training.py --config_file=env_configs/multi_agent/a2c_training.yml
python rl_training.py --config_file=env_configs/multi_agent/td3_training.yml
```

# Citing the Project
If you use this package, please cite the following work:
``` 
@misc{lavington2024torchdriveenv,
      title={TorchDriveEnv: A Reinforcement Learning Benchmark for Autonomous Driving with Reactive, Realistic, and Diverse Non-Playable Characters}, 
      author={Jonathan Wilder Lavington and Ke Zhang and Vasileios Lioutas and Matthew Niedoba and Yunpeng Liu and Dylan Green and Saeid Naderiparizi and Xiaoxuan Liang and Setareh Dabiri and Adam Ścibior and Berend Zwartsenberg and Frank Wood},
      year={2024},
      eprint={2405.04491},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
