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

wget https://files.pythonhosted.org/packages/79/4a/a0697639ea1f0de179b52b3698b0ea3ffa7fe1e7e8f393c3acda3a4730da/lanelet2-1.2.1-cp38-cp38-manylinux_2_27_x86_64.whl
mv lanelet2-1.2.1-cp38-cp38-manylinux_2_27_x86_64.whl lanelet2-1.2.1-cp38-none-any.whl
wget https://files.pythonhosted.org/packages/a6/80/0f2fb0deada2fbfadfec6b4dd1f5536767a4eb66dc3b7bd46539a1e232a0/pydantic_core-2.18.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
mv pydantic_core-2.18.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl pydantic_core-2.18.2-cp38-none-any.whl

pip install lanelet2-1.2.1-cp38-none-any.whl
pip install pydantic_core-2.18.2-cp38-none-any.whl

pip3 install torchdriveenv[baselines] --cert /etc/ssl/certs/ca-bundle.crt

export IAI_API_KEY=oOIwcHIF0v4g4t7awNvEk6QObz7eflCq9MCCQNLQ
export WANDB_API_KEY=8160f073728588c7ce58759b1b24ce284b05705c

python rl_training.py
