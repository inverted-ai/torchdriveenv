# Installation

## Opencv rendering

To install the “torchdriveenv” with opencv rendering:
```
python3 -m venv $PHTHON_VIRTUAL_ENV_PATH
source .venv/bin/activate
pip install "torchdrivesim @ git+https://github.com/inverted-ai/torchdrivesim.git@first-release-env"
pip install "torchdriveenv[baselines] @ git+ssh://git@github.com/inverted-ai/torchdriveenv.git@first-release-env"
```

To run examples:
Set the `$IAI_API_KEY` and `$WANDB_API_KEY`
```
git clone git@github.com:inverted-ai/torchdriveenv.git
cd torchdriveenv
git checkout first-release-env
cd examples
source $PHTHON_VIRTUAL_ENV_PATH/bin/activate
python rl_training.py
```

## Pytorch3d rendering

To install the “torchdriveenv” with Pytorch3d rendering:
```
git clone git@github.com:inverted-ai/torchdriveenv.git
cd torchdriveenv
git checkout first-release-env
docker build --target torchdriveenv-first-release -t torchdriveenv-first-release:latest .
```

To run examples:
Set the `$IAI_API_KEY` and `$WANDB_API_KEY`
```
cd torchdriveenv
git checkout first-release-env
cd examples
docker compose up rl-training
```
