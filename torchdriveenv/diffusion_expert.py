import torch
import numpy as np
from scipy.ndimage import zoom
from edm.pl_modules import EDMModule

class DiffusionExpert:
    def __init__(self, pretrained_module_path):
        self.module = EDMModule.load_from_checkpoint(pretrained_module_path)

    def expert_prob(self, action, observation):
        with torch.no_grad():
            logp = self.module.diffusion.logpx(x_0=action.unsqueeze(0), s=observation.unsqueeze(0))[0]
        return logp

    def expert_logp_from_energy(self, action, observation):
        with torch.no_grad():
            t = 0

            nticks = 10

    #        vmin_x = -0.3
            vmin_x = -3.0
            vmin_y = -1.0
    #        vmax_x = 0.3
            vmax_x = 3.0
            vmax_y = 1.0

            steering, acceleration = np.meshgrid(np.linspace(vmin_x, vmax_x, nticks), np.linspace(vmin_y, vmax_y, nticks))
            steering = torch.Tensor(steering)
            acceleration = torch.Tensor(acceleration)
            actions = torch.stack([acceleration, steering], axis=-1).reshape(-1, 2).to(observation.device)
            t = torch.Tensor([t]).int().expand((actions.shape[0], )).to(observation.device)
            observation = observation.unsqueeze(0).expand((actions.shape[0], *observation.shape)).to(observation.device)

            log_p = - self.module.diffusion.p_energy(x=actions, t=t, s=observation).reshape(
                (nticks, nticks)).detach().cpu().transpose(0, 1).numpy()

            log_p_max = np.max(log_p)
            p = np.exp(log_p - log_p_max)  # Subtract the max log probability and exponentiate
            p /= np.sum(p)
#            p *= 1024.0
#            p = np.clip(p, 0, 1.0)

            interpolated_p = torch.Tensor(zoom(p, (10, 10), order=1)).to(observation.device)

            index_x = torch.round((action[1] * 3.0 - vmin_x) / (vmax_x - vmin_x) * 100).long()
            index_y = torch.round((action[0] - vmin_y) / (vmax_y - vmin_y) * 100).long()

            if index_x == 100:
                index_x -= 1
            if index_y == 100:
                index_y -= 1

        return torch.log(interpolated_p[index_x, index_y])

    def expert_batch_logp_from_energy(self, action, observation):
        logp = []
        for i in range(action.shape[0]):
            logp.append(self.expert_logp_from_energy(action[i], observation[i].float()))
        return torch.stack(logp)

    def expert_energy(self, t, action, observation):
        with torch.no_grad():
            energy = self.module.diffusion.p_energy(x_0=action.unsqueeze(0), t=t, s=observation.unsqueeze(0))
        return energy
