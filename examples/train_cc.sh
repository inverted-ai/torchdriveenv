#!/bin/bash
#SBATCH --account=rrg-fwood
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=16
#SBATCH --time=0-02:00:00

module load StdEnv/2020
module load python/3.8.10
module load boost/1.80.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

python -m pip install --upgrade pip
pip install torchdriveenv[baselines]

export IAI_API_KEY=oOIwcHIF0v4g4t7awNvEk6QObz7eflCq9MCCQNLQ
export WANDB_API_KEY=8160f073728588c7ce58759b1b24ce284b05705c

python rl_training.py
