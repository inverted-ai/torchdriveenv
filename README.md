# Overview
TorchDriveEnv is a lightweight, permissiviely licensed reinforcement learning benchmark for autonomous driving, that provides human like non-playable characters (NPCs) and is easy to set up and use. The benchmark provides NPCs through the state of the art behavioral models provided through the Inverted AI API (link). Torchdriveenv is quick to set up and comes integrated with stablebaselines3 (link), making the time invested in getting started low. Nevertheless, it is fully customizable, as new scenarios are easy to define and modifying the reward is simple. Moreover, since it is built on TorchDriveSim (link), it comes with various kinematic models, agent types and traffic controls. 

# Reward
Torchdriveenv is based on waypoints that are provided to the ego agent, and a reward for movement, smoothness and number of achieved waypoints. Episodes that cause infractions (causing collisions or going offroad) are terminated. For a full overview of the reward, refer to the paper linked below. 

# Scenario overview
Torchdriveenv comes bundled with a predefined collection of diverse and challenging training and testing scenarios. Training scenarios were created using the InvertedAI API, while the testing scenarios were hand tailored for interesting challenges. An overview of the testing scenarios is given in the table below, as well as videos demonstrating baseline performance. Note that while the ego agent has fixed waypoints, its starting position is sampled uniformly between the first two waypoints, and NPCs are initialized dynamically using the InvertedAI API. This causes a large variation in the base scenarios. 

# Installation

The basic installation of torchdriveenv uses an OpenCV renderer, which is slower but easy to install. PyTorch3D renderer can be faster, but it requires specific versions of CUDA and PyTorch, so it is best installed using Docker.

## Opencv rendering

To install the “torchdriveenv” with opencv rendering:
```
pip install torchdriveenv
```

To run the example:
```
pip install torchdriveenv[baselines]
cd examples
python rl_training.py
```

## Pytorch3d rendering

To install torchdriveenv with Pytorch3d rendering, we recommend an installation through docker. A Dockerfile is provided and can be built using:
```
docker build --target torchdriveenv-first-release -t torchdriveenv-first-release:latest .
```

To run the example:
```
cd examples
docker compose up rl-training
```
## NPCs
To use the NPC's provided through the InvertedAI API, provide a valid API key through the environment variable `IAI_API_KEY`

## Wandb
TorchDriveEnv is integrated with Weights and Biasses for experiment logging, to make use of it provide an API key through `WANDB_API_KEY`

## Running experiments
The results reported in the paper can be reproduced by running:

``` 
TODO
```

# Citation
If you use this package, please cite the following work:

``` 


```
