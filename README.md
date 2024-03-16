# Installation

## Install without Pytorch3d

```
python3 -m venv .venv
source .venv/bin/activate
pip install torchdrivesim git+https://github.com/inverted-ai/torchdrivesim.git@cleanup
pip install "torchdriveenv[baselines] @ git+ssh://git@github.com/inverted-ai/rl-env.git@cleanup"
```

## Setup the repo

1. Run `git checkout add_env` and `git pull` to pull the `add_env` branch of this repo.

2. Run `git submodule update --init` to get the `torchdrivesim`.

## Python virtual env

1. Ensure cuda version is 11.6

2. Create virtual env

3. `pip install -r requirements.txt`

## Docker

1. `cd torchdrivesim`, run `docker build --target rl-env-base -t rl-env-base:latest .` to build the container.

2. `cd ..`, `cp api_key_template .env`, set the variables in the `.env` to your own keys.

3. `docker compose up notebook` to start the jupyter notebook.

4. Open the jupyter notebook in your web browser, under the `/opt/rl-env/` there are three notebooks, `task_env_example.ipynb` to visualize the RL-env itself, `rl_example.ipynb` to train the RL models and `evaluation` to evaluate a trained model and generate a global birdview video.



