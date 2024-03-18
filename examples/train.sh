#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00

module load python/3.8.18
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python
which python
which pip

pip3 install -r requirements.txt

export IAI_API_KEY=
export WANDB_API_KEY=

python rl_training.py
