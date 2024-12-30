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


class VisualizeEvaluationCallback(BaseCallback):
    def __init__(self, eval_data_dirs, eval_freq=100, sample_num=3, verbose=0):
        super(VisualizeEvaluationCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.sample_num = sample_num
        self._load_evaluation_datasets(eval_data_dirs)

    def _load_evaluation_datasets(self, data_dirs):
        self.eval_datasets = []
        for data_dir in data_dirs:
            self.eval_datasets.append(load_pickle_dataset(data_dir))

    def _get_device(self):
        return next(self.model.critic.parameters()).device

    def _visualize_critic(self):
        device = self._get_device()
        for dataset in self.eval_datasets:
            for i, obs in enumerate(random.choices(dataset, k=self.sample_num)):
                obs_image = Image.fromarray(obs[:3].cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0), 'RGB')
                critic_heatmap = plot_critic(self.model.critic.q1_forward, obs, device)
                wandb.log({
                        f"critic visualization for sample {i}": [wandb.Image(obs_image, caption="observation"), wandb.Image(critic_heatmap, caption="critic")]
                })

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.model.critic.eval()
            with torch.no_grad():
                self._visualize_critic()
            self.model.critic.train()

        return True  # Continue training
