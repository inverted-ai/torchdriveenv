import io
import os
import cv2
import pickle
import torch
import wandb
import numpy as np

from PIL import Image, ImageOps
from matplotlib import pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_grid_action_prob

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


def to_video(pil_images, video_name="video.mp4", fps=10):

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


def resize_and_pad_image(image, target_width, target_height):
    if image.width < target_width / 2:
        scale_width = target_width / image.width
        scale_height = target_height / image.height

        scale_factor = min(scale_width, scale_height)

        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)

        image = image.resize((new_width, new_height), Image.LANCZOS)

    delta_width = target_width - image.width
    delta_height = target_height - image.height
    padding = (delta_width // 2, delta_height // 2, delta_width -
               delta_width // 2, delta_height - delta_height // 2)

    return ImageOps.expand(image, padding, fill=(255, 255, 255))


def concat_images_2x2(images):
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    padded_images = [resize_and_pad_image(
        img, max_width, max_height) for img in images]

    grid_width = max_width * 2
    grid_height = max_height * 2
    grid_image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

    grid_image.paste(padded_images[0], (0, 0))
    grid_image.paste(padded_images[1], (max_width, 0))
    grid_image.paste(padded_images[2], (0, max_height))
    grid_image.paste(padded_images[3], (max_width, max_height))

    return grid_image


def plot_heatmap(heatmap, name="critic"):
    vmin_x = -1.0
    vmin_y = -1.0
    vmax_x = 1.0
    vmax_y = 1.0

    heatmap = torch.flip(heatmap.squeeze(), dims=[1]).detach().cpu().numpy()

    plt.imshow(heatmap, extent=[vmin_x, vmax_x, vmin_y,
               vmax_y], origin='lower', cmap='viridis')

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.colorbar(label=name)
    plt.xlabel("steering")
    plt.ylabel("acceleration")
    return to_image(plt)


def plot_normal_dist(mean, stddev, name="current policy model"):
    dist = torch.distributions.Normal(mean, stddev)

    nticks = 100
    vmin_x = -1.0
    vmin_y = -1.0
    vmax_x = 1.0
    vmax_y = 1.0

    x = np.linspace(vmin_x, vmax_x, nticks)
    y = np.linspace(vmin_y, vmax_y, nticks)
    X, Y = np.meshgrid(x, y)

    grid = torch.tensor(
        np.stack([X, Y], axis=2).reshape(-1, 2), dtype=torch.float32).to(mean.device)

    pdf_values = torch.flip(torch.exp(dist.log_prob(
        grid).sum(-1)).reshape(100, 100).transpose(0, 1), dims=[1]).cpu().detach().numpy()

    plt.imshow(pdf_values, extent=[
               vmin_x, vmax_x, vmin_y, vmax_y], origin='lower', cmap='viridis')

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.title(name)
    plt.colorbar(label="probability")
    plt.xlabel("steering")
    plt.ylabel("acceleration")

    return to_image(plt)


class EvalRolloutCallback(BaseCallback):
    def __init__(self, eval_env, rollout_episode_num=1, verbose=0):
        super(EvalRolloutCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.rollout_episode_num = rollout_episode_num
        self.diffusion_expert = DiffusionExpert(
            "pretrained_edm_module/model.ckpt")

    def _on_step(self) -> bool:
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
            stacked_obs_tensor = torch.Tensor(stacked_obs).squeeze().to(device)
            mean, log_std, _ = self.model.policy.actor.get_action_dist_params(
                stacked_obs_tensor.unsqueeze(0))
            policy_plots = [plot_normal_dist(
                mean.squeeze(), torch.exp(log_std.squeeze()))]
#            b_critic_plots = [plot_heatmap(self.model.get_b_grid(
#                stacked_obs_tensor), name="b_critic")]
            b_critic_plots = [plot_heatmap(get_grid_action_prob(fn=self.model.critic_target,
                                                                observation=stacked_obs_tensor),
                                           name="b_critic")]
#            c_critic_plots = [plot_heatmap(self.model.get_c_grid(
#                stacked_obs_tensor), name="c_critic")]
            c_critic_plots = [plot_heatmap(get_grid_action_prob(fn=self.diffusion_expert.module.diffusion.p_energy,
                                                                observation=stacked_obs_tensor, p_energy=True),
                                           name="c_critic")]

            while not done:
                action, _ = self.model.predict(stacked_obs, deterministic=True)

                obs, reward, done, info = self.eval_env.step(action)
                obs_list.append(obs)
                stacked_obs = np.concatenate(obs_list[-3:], axis=-3)
                stacked_obs_tensor = torch.Tensor(
                    stacked_obs).squeeze().to(device)
                mean, log_std, _ = self.model.policy.actor.get_action_dist_params(
                    stacked_obs_tensor.unsqueeze(0))
                policy_plots.append(plot_normal_dist(
                    mean.squeeze(), torch.exp(log_std.squeeze())))
                b_critic_plots.append(plot_heatmap(get_grid_action_prob(fn=self.model.critic_target,
                                                                        observation=stacked_obs_tensor),
                                                   name="b_critic"))
                c_critic_plots.append(plot_heatmap(get_grid_action_prob(fn=self.diffusion_expert.module.diffusion.p_energy,
                                                                        observation=stacked_obs_tensor, p_energy=True),
                                                   name="c_critic"))
#                b_critic_plots.append(plot_heatmap(
#                    self.model.get_b_grid(stacked_obs_tensor), name="b_critic"))
#                c_critic_plots.append(plot_heatmap(
#                    self.model.get_c_grid(stacked_obs_tensor), name="c_critic"))

            videos = []
            obs_images = [Image.fromarray(obs.squeeze().astype(
                np.uint8).transpose(1, 2, 0), 'RGB') for obs in obs_list[2:]]
            concat_images = [concat_images_2x2(
                [obs_images[i], policy_plots[i], b_critic_plots[i], c_critic_plots[i]]) for i in range(len(obs_images))]

            videos.append(wandb.Video(to_video(concat_images)))
            wandb.log({f"rollout from current policy": videos})
