import io
import os
import pickle
import random
import torch
import wandb
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback

from torchdriveenv.diffusion_expert import DiffusionExpert


def load_pickle_dataset(data_dir):
    dataset = []
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if file_path[-4:] != ".pkl":
            continue
        with open(file_path, "rb") as f:
            obs_data = pickle.load(f)
            dataset.append(obs_data)
    return dataset


def to_image(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def plot_critic(fn, obs, device):
    nticks = 50

#    vmin_x = -2
#    vmin_y = -2
#    vmax_x = 2
#    vmax_y = 2

    vmin_x = -1.0
    vmin_y = -1.0
    vmax_x = 1.0
    vmax_y = 1.0


    steering, acceleration = np.meshgrid(np.linspace(vmin_x, vmax_x, nticks), np.linspace(vmin_y, vmax_y, nticks))
    steering = torch.Tensor(steering)
    acceleration = torch.Tensor(acceleration)
    actions = torch.stack([acceleration, steering], axis=-1).reshape(-1, 2).to(device)
    obs = obs.unsqueeze(0).expand((actions.shape[0], *obs.shape)).to(device)

    heatmap = fn(actions=actions, obs=obs).reshape(
        (nticks, nticks)).detach().cpu().transpose(0, 1).numpy()

    plt.imshow(heatmap, extent=[vmin_x, vmax_x, vmin_y, vmax_y], origin='lower', cmap='viridis')

    plt.axhline(0, color='black', linewidth=1)  # Horizontal axis
    plt.axvline(0, color='black', linewidth=1)  # Vertical axis

    # Add labels and color bar
    plt.colorbar(label="critic")
    plt.xlabel("steering")
    plt.ylabel("acceleration")
    return to_image(plt)


def plot_elbo(fn, obs, device):
    nticks = 10

    vmin_x = -2
    vmin_y = -2
    vmax_x = 2
    vmax_y = 2

    steering, acceleration = np.meshgrid(np.linspace(vmin_x, vmax_x, nticks), np.linspace(vmin_y, vmax_y, nticks))
    steering = torch.Tensor(steering)
    acceleration = torch.Tensor(acceleration)
    actions = torch.stack([acceleration, steering], axis=-1).reshape(-1, 2).to(device)
#    t = torch.Tensor([t]).int().expand((actions.shape[0], )).to(device)
    obs = obs.unsqueeze(0).expand((actions.shape[0], *obs.shape)).to(device)

    heatmap = fn(x_0=actions, s=obs).reshape(
        (nticks, nticks, 100))[:, :, 0].detach().cpu().transpose(0, 1).numpy()
    plt.imshow(heatmap, extent=[vmin_x, vmax_x, vmin_y, vmax_y], origin='lower', cmap='viridis')

    plt.axhline(0, color='black', linewidth=1)  # Horizontal axis
    plt.axvline(0, color='black', linewidth=1)  # Vertical axis

    # Add labels and color bar
    plt.colorbar(label="logp via ELBO")
    plt.xlabel("steering")
    plt.ylabel("acceleration")
    return to_image(plt)


def plot_energy(fn, obs, device):
    t = 0
    nticks = 50

    vmin_x = -2
    vmin_y = -2
    vmax_x = 2
    vmax_y = 2

    steering, acceleration = np.meshgrid(np.linspace(vmin_x, vmax_x, nticks), np.linspace(vmin_y, vmax_y, nticks))
    steering = torch.Tensor(steering)
    acceleration = torch.Tensor(acceleration)
    actions = torch.stack([acceleration, steering], axis=-1).reshape(-1, 2).to(device)
    t = torch.Tensor([t]).int().expand((actions.shape[0], )).to(device)
    obs = obs.unsqueeze(0).expand((actions.shape[0], *obs.shape)).to(device)

    heatmap = (100000 - fn(x=actions, t=t, s=obs)).reshape(
        (nticks, nticks)).detach().cpu().transpose(0, 1).numpy()
    plt.imshow(heatmap, extent=[vmin_x, vmax_x, vmin_y, vmax_y], origin='lower', cmap='viridis')

    plt.axhline(0, color='black', linewidth=1)  # Horizontal axis
    plt.axvline(0, color='black', linewidth=1)  # Vertical axis

    # Add labels and color bar
    plt.colorbar(label="unnormalized log likelihood")
    plt.xlabel("steering")
    plt.ylabel("acceleration")
    return to_image(plt)


class VisualizeEvaluationCallback(BaseCallback):
    def __init__(self, eval_data_dirs, eval_freq=100, sample_num=3, verbose=0):
        super(VisualizeEvaluationCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.sample_num = sample_num
        self._load_evaluation_datasets(eval_data_dirs)
        self.diffusion_expert = DiffusionExpert("pretrained_edm_module/model.ckpt")

    def _load_evaluation_datasets(self, data_dirs):
        self.eval_datasets = []
        for data_dir in data_dirs:
            self.eval_datasets.append(load_pickle_dataset(data_dir))

    def _get_device(self):
        return next(self.model.critic.parameters()).device

    def _visualize(self, online_critic=True, offline_elbo=False, offline_energy=False, offline_critic=False):
        device = self._get_device()
        for dataset in self.eval_datasets:
            for i, obs in enumerate(random.choices(dataset, k=self.sample_num)):
                obs_image = Image.fromarray(obs[:3].cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0), 'RGB')
                images = [wandb.Image(obs_image, caption="Observation")]
                if online_critic:
                    images.append(wandb.Image(plot_critic(self.model.critic.q1_forward, obs, device), caption="Online Critic"))
                if offline_elbo:
                    images.append(wandb.Image(plot_elbo(self.diffusion_expert.module.diffusion.logpx, obs, device), caption="Offline ELBO"))
                if offline_energy:
                    images.append(wandb.Image(plot_energy(self.diffusion_expert.module.diffusion.p_energy, obs, device), caption="Offline Energy"))
#                if offline_critic:
#                    images.append(wandb.Image(plot_critic(self.training_env.offline_critic.q1_critic, obs, device), caption="Offline Critic"))
                wandb.log({f"Sample {i}": images})

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.model.critic.eval()
            with torch.no_grad():
                self._visualize(online_critic=True, offline_elbo=True, offline_energy=True, offline_critic=False)
            self.model.critic.train()

        return True  # Continue training
