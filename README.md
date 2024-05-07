# Overview
TorchDriveEnv is a lightweight reinforcement learning benchmark for autonomous driving written entirely in python, that provides human like non-playable characters (NPCs). Installation is as easy as running:

```
pip install torchdriveenv[baselines]
```

The benchmark provides NPCs through the state of the art behavioral models provided through the [Inverted AI API](https://www.inverted.ai/home). While emphasis is on ease of use, TorchDriveEnv is fully customizable, with easy to define new scenarios or reward. TorchDriveEnv is built on the lightweight driving simulator [TorchDriveSim](https://github.com/inverted-ai/torchdrivesim/), and therefore comes with various kinematic models, agent types and traffic controls. 


## Scenario overview
TorchDriveEnv comes bundled with a predefined collection of diverse and challenging training and testing scenarios. Training scenarios were created using the InvertedAI API, while the testing scenarios were hand tailored for interesting challenges. An overview of the testing scenarios is given in the visualization below. Note that while the ego agent has fixed waypoints in these scenarios, its starting position is sampled uniformly between the first two waypoints, and NPCs are initialized dynamically using the InvertedAI API. This results in a large variation in the available scenarios. 

## Reward
The reward for TorchDriveEnv is based on waypoints that are provided to the ego agent, as well as a reward for movement and smoothness. Episodes that cause infractions (causing collisions or going offroad) are terminated early, otherwise the episode is terminated after a fixed number of steps. For a full overview of the reward, refer to the paper linked below. 

# Installation

The basic installation of torchdriveenv uses an OpenCV renderer, which is slower but easy to install. PyTorch3D renderer can be faster, but it requires specific versions of CUDA and PyTorch, so it is best installed using Docker.

TorchDriveEnv comes with an example that integrates [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master), which can be used for testing purposes.

## Opencv rendering

To install the “torchdriveenv” with opencv rendering it suffices to simply:
```
pip install torchdriveenv[baselines]
```
An example can be accessed by first cloning this repository and then:
```
cd examples
python rl_training.py
```

## Pytorch3d rendering

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
To use the NPC's provided through the InvertedAI API, provide a valid API key through the environment variable `IAI_API_KEY`.

## Wandb
TorchDriveEnv is integrated with Weights & Biases for experiment logging, to make use of it provide an API key through `WANDB_API_KEY`.

## Running experiments
The results reported in the paper can be reproduced by cloning this repository and running:

``` 
python rl_training.py --config_file=env_configs/multi_agent/sac_training.yml
python rl_training.py --config_file=env_configs/multi_agent/ppo_training.yml
python rl_training.py --config_file=env_configs/multi_agent/a2c_training.yml
python rl_training.py --config_file=env_configs/multi_agent/td3_training.yml
```

# Citation
If you use this package, please cite the following work:
``` 


```
