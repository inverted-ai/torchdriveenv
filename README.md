# Installation

The basic installation of torchdriveenv uses an OpenCV renderer, which is slower but easy to install. PyTorch3D renderer can be faster, but it requires specific versions of CUDA and PyTorch, so it is best installed in Docker.

## Opencv rendering

To install the “torchdriveenv” with opencv rendering:
```
pip install torchdriveenv
```

To run examples:
Set the `$IAI_API_KEY` and `$WANDB_API_KEY`
```
pip install torchdriveenv[baselines]
cd examples
python rl_training.py
```

## Pytorch3d rendering

To install the “torchdriveenv” with Pytorch3d rendering:
```
docker build --target torchdriveenv-first-release -t torchdriveenv-first-release:latest .
```

To run examples:
Set the `$IAI_API_KEY` and `$WANDB_API_KEY`
```
cd examples
docker compose up rl-training
```
