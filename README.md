# Installation

The basic installation of torchdriveenv uses an OpenCV renderer, which is slower but easy to install. PyTorch3D renderer can be faster, but it requires specific versions of CUDA and PyTorch, so it is best installed in Docker.

## Opencv rendering

To install the “torchdriveenv” with opencv rendering:
```
python3 -m venv $PHTHON_VIRTUAL_ENV_PATH
source .venv/bin/activate
pip install "torchdrivesim @ git+https://github.com/inverted-ai/torchdrivesim.git@first-release-env"
pip install "torchdriveenv[baselines] @ git+https://github.com/inverted-ai/torchdriveenv.git"
```

To run examples:
Set the `$IAI_API_KEY` and `$WANDB_API_KEY`
```
git clone git@github.com:inverted-ai/torchdriveenv.git
cd torchdriveenv
cd examples
source $PHTHON_VIRTUAL_ENV_PATH/bin/activate
python rl_training.py
```

## Pytorch3d rendering

To install the “torchdriveenv” with Pytorch3d rendering:
```
git clone git@github.com:inverted-ai/torchdriveenv.git
cd torchdriveenv
docker build --target torchdriveenv-first-release -t torchdriveenv-first-release:latest .
```

To run examples:
Set the `$IAI_API_KEY` and `$WANDB_API_KEY`
```
cd torchdriveenv
cd examples
docker compose up rl-training
```
