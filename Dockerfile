FROM nvidia/cuda:11.7.1-devel-ubuntu20.04 as torchdriveenv-first-release
# This file is for building the production api server only.

# Install general utilities
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y tzdata software-properties-common && add-apt-repository universe && apt update && apt install -y locales-all  git curl sed nano

# Install Python and related tooling
RUN apt install -y libffi-dev python python-dev python3-dev python3-pip python3-venv \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
  && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
  && pip install setuptools distro wheel \
  && apt install -y build-essential llvm libxml2-dev

RUN apt install -y nvidia-modprobe

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/root/.bash_history" \
    && echo "$SNIPPET" >> "/root/.bashrc"

# Install pytorch and pytorch3d
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html

# Set up C compilers and build systems
RUN apt update && apt install -y build-essential cmake autoconf ninja-build \
  clang-8 lld-8 g++-7 \
  libpng-dev libtiff5-dev libjpeg-dev libtool libxml2-dev libxerces-c-dev libgl1

RUN pip install -U pip
RUN pip install lanelet2
RUN pip install opencv-python
RUN pip install gymnasium[classic-control] stable-baselines3
RUN pip install wandb tensorboard moviepy
RUN pip install jupyter notebook
RUN pip install omegaconf scipy shapely
RUN pip install invertedai==0.0.14
RUN pip install "torchdrivesim @ git+https://github.com/inverted-ai/torchdrivesim.git@first-release-env"
RUN pip install "torchdriveenv[baselines] @ git+https://github.com/inverted-ai/torchdriveenv.git@first-release-env"
