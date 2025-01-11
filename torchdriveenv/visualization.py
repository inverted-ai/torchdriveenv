import io
import os
import cv2
import pickle
import random
import torch
import wandb
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import zoom

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import check_static

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


def to_video(pil_images, video_name = "video.mp4", fps=10):

    frames = [np.array(img) for img in pil_images]

    height, width, layers = frames[0].shape
    size = (width, height)


    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(video_name,
                               fourcc, fps, size)

    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

    video_writer.release()
    return video_name


def to_gif(pil_images, gif_name=None, duration=100):
    save_to = gif_name if gif_name is not None else io.BytesIO()
    pil_images[0].save(save_to,
                       format="GIF",
                       save_all=True,
                       append_images=pil_images[1:],
                       duration=duration,
                       loop=0)
    return save_to


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
        (nticks, nticks)).transpose(0, 1).detach().cpu().numpy()

    plt.imshow(heatmap, extent=[vmin_x, vmax_x, vmin_y, vmax_y], origin='lower', cmap='viridis')

    plt.axhline(0, color='black', linewidth=1)  # Horizontal axis
    plt.axvline(0, color='black', linewidth=1)  # Vertical axis

    # Add labels and color bar
    plt.colorbar(label="critic")
    plt.xlabel("steering")
    plt.ylabel("acceleration")
    return to_image(plt)


def plot_heatmap(heatmap, name="critic"):
    vmin_x = -1.0
    vmin_y = -1.0
    vmax_x = 1.0
    vmax_y = 1.0

    heatmap = torch.flip(heatmap.squeeze(), dims=[1]).detach().cpu().numpy()
#    heatmap = heatmap.squeeze().detach().cpu().numpy()
#    heatmap = torch.flip(heatmap.squeeze().transpose(0, 1), dims=[1]).detach().cpu().numpy()

    plt.imshow(heatmap, extent=[vmin_x, vmax_x, vmin_y, vmax_y], origin='lower', cmap='viridis')

    plt.axhline(0, color='black', linewidth=1)  # Horizontal axis
    plt.axvline(0, color='black', linewidth=1)  # Vertical axis

    # Add labels and color bar
    plt.colorbar(label=name)
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


def plot_normalized_energy(fn, obs, device):
    t = 0
    nticks = 10

    vmin_x = -3.0
    vmin_y = -1.0
#        vmax_x = 0.3
    vmax_x = 3.0
    vmax_y = 1.0

    steering, acceleration = np.meshgrid(np.linspace(vmin_x, vmax_x, nticks), np.linspace(vmin_y, vmax_y, nticks))
    steering = torch.Tensor(steering)
    acceleration = torch.Tensor(acceleration)
    actions = torch.stack([acceleration, steering], axis=-1).reshape(-1, 2).to(device)
    t = torch.Tensor([t]).int().expand((actions.shape[0], )).to(device)
    obs = obs.unsqueeze(0).expand((actions.shape[0], *obs.shape)).to(device)
    log_p = - fn(x=actions, t=t, s=obs).reshape(
        (nticks, nticks)).detach().cpu().transpose(0, 1).numpy()

    log_p_max = np.max(log_p)
    p = np.exp(log_p - log_p_max)  # Subtract the max log probability and exponentiate
    p /= np.sum(p)

    interpolated_p = zoom(p, (10, 10), order=1)
#
#    heatmap = - fn(x=actions, t=t, s=obs).reshape(
#        (nticks, nticks)).detach().cpu().transpose(0, 1).numpy()

#    scaled_heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
#    normalized_heatmap = scaled_heatmap / scaled_heatmap.sum()
#
#    interpolated_heatmap = zoom(normalized_heatmap, 10, order=1)

    plt.imshow(interpolated_p, extent=[vmin_x, vmax_x, vmin_y, vmax_y], origin='lower', cmap='viridis')

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


class EvalRolloutCallback(BaseCallback):
    def __init__(self, eval_env, rollout_episode_num=1, verbose=0):
        super(EvalRolloutCallback, self).__init__(verbose)
        self.eval_env = eval_env  # Evaluation environment
        self.rollout_episode_num = rollout_episode_num
        self.diffusion_expert = DiffusionExpert("pretrained_edm_module/model.ckpt")

    def _on_step(self) -> bool:
        # Perform a rollout every 1000 steps (as an example)
        if self.n_calls % 100 == 0:
            with torch.no_grad():
                self.perform_rollout()
        return True

    def _get_device(self):
        return next(self.model.critic.parameters()).device

    def perform_rollout(self):
        """
        Perform a rollout in the evaluation environment with the current policy.
        """
        device = self._get_device()

        for episode in range(self.rollout_episode_num):
            obs = self.eval_env.reset()
            done = False
            obs_list = [obs, obs, obs]
            stacked_obs = np.concatenate(obs_list[-3:], axis=-3)
            online_critic_plots = [plot_critic(self.model.critic.q1_forward, torch.Tensor(stacked_obs).squeeze(), device)]
#            offline_elbo_plots = [plot_elbo(self.diffusion_expert.module.diffusion.logpx, torch.Tensor(stacked_obs).squeeze(), device)]
            offline_energy_plots = [plot_energy(self.diffusion_expert.module.diffusion.p_energy, torch.Tensor(stacked_obs).squeeze(), device)]
            offline_normalized_energy_plots = [plot_normalized_energy(self.diffusion_expert.module.diffusion.p_energy, torch.Tensor(stacked_obs).squeeze(), device)]
            while not done:
                action, _ = self.model.predict(stacked_obs, deterministic=True)

                obs, reward, done, info = self.eval_env.step(action)
                obs_list.append(obs)
                stacked_obs = np.concatenate(obs_list[-3:], axis=-3)
                online_critic_plots.append(plot_critic(self.model.critic.q1_forward, torch.Tensor(stacked_obs).squeeze(), device))
#                offline_elbo_plots.append(plot_elbo(self.diffusion_expert.module.diffusion.logpx, torch.Tensor(stacked_obs).squeeze(), device))
                offline_energy_plots.append(plot_energy(self.diffusion_expert.module.diffusion.p_energy, torch.Tensor(stacked_obs).squeeze(), device))
                offline_normalized_energy_plots.append(plot_normalized_energy(self.diffusion_expert.module.diffusion.p_energy, torch.Tensor(stacked_obs).squeeze(), device))

            videos = []
            obs_video = to_video([Image.fromarray(obs.squeeze().astype(np.uint8).transpose(1, 2, 0), 'RGB') for obs in obs_list], video_name="obs.mp4")
            videos.append(wandb.Video(obs_video, caption="Observation"))
            online_critic_video = to_video(online_critic_plots, video_name="online_critic.mp4")
            videos.append(wandb.Video(online_critic_video, caption="Online Critic"))
#            offline_elbo_video = to_video(offline_elbo_plots)
            offline_energy_video = to_video(offline_energy_plots, video_name="offline_energy.mp4")
            videos.append(wandb.Video(offline_energy_video, caption="Offline Energy"))
            offline_normalized_energy_video = to_video(offline_normalized_energy_plots, video_name="offline_normalized_energy.mp4")
            videos.append(wandb.Video(offline_normalized_energy_video, caption="Offline Normalized Probability from Energy"))
#            videos.append(wandb.Video(offline_elbo_video, caption="Offline ELBO"))
            wandb.log({f"rollout from current policy": videos})


class EvalWABCCallback(BaseCallback):
    def __init__(self, eval_env, rollout_episode_num=1, verbose=0):
        super(EvalWABCCallback, self).__init__(verbose)
        self.eval_env = eval_env  # Evaluation environment
        self.rollout_episode_num = rollout_episode_num
        self.diffusion_expert = DiffusionExpert("pretrained_edm_module/model.ckpt")

    def _on_step(self) -> bool:
        # Perform a rollout every 1000 steps (as an example)
        if self.n_calls % 100 == 0:
            with torch.no_grad():
                self.perform_rollout()
        return True

    def _get_device(self):
        return next(self.model.b_net.parameters()).device

    def perform_rollout(self):
        """
        Perform a rollout in the evaluation environment with the current policy.
        """
        device = self._get_device()

        for episode in range(self.rollout_episode_num):
            obs = self.eval_env.reset()
            done = False
            obs_list = [obs, obs, obs]
            stacked_obs = np.concatenate(obs_list[-3:], axis=-3)
            stacked_obs_tensor = torch.Tensor(stacked_obs).squeeze().to(device)
            w_critic_plots = [plot_heatmap(self.model.policy.get_w_grid(stacked_obs_tensor))]
            b_critic_plots = [plot_heatmap(self.model.policy.get_b_grid(stacked_obs_tensor))]
            c_critic_plots = [plot_heatmap(self.model.policy.get_c_grid(stacked_obs_tensor))]

            while not done:
                action, _ = self.model.predict(stacked_obs, deterministic=False)

                obs, reward, done, info = self.eval_env.step(action)
#                print("check_static")
#                print(check_static(stacked_obs_tensor.unsqueeze(0)))
#                print("action in perform_rollout")
#                print(action)
                obs_list.append(obs)
                stacked_obs = np.concatenate(obs_list[-3:], axis=-3)
                stacked_obs_tensor = torch.Tensor(stacked_obs).squeeze().to(device)
                w_critic_plots.append(plot_heatmap(self.model.policy.get_w_grid(stacked_obs_tensor)))
                b_critic_plots.append(plot_heatmap(self.model.policy.get_b_grid(stacked_obs_tensor)))
                c_critic_plots.append(plot_heatmap(self.model.policy.get_c_grid(stacked_obs_tensor)))

            videos = []
            obs_video = to_video([Image.fromarray(obs.squeeze().astype(np.uint8).transpose(1, 2, 0), 'RGB') for obs in obs_list], video_name="obs.mp4")
            videos.append(wandb.Video(obs_video, caption="Observation"))
            w_critic_video = to_video(w_critic_plots, video_name="w_critic.mp4")
            videos.append(wandb.Video(w_critic_video, caption="W Critic"))
            b_critic_video = to_video(b_critic_plots, video_name="b_critic.mp4")
            videos.append(wandb.Video(b_critic_video, caption="B Critic"))
            c_critic_video = to_video(c_critic_plots, video_name="c_critic.mp4")
            videos.append(wandb.Video(c_critic_video, caption="C Critic"))
            wandb.log({f"rollout from current policy": videos})
