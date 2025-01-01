import torch
from edm.pl_modules import EDMModule

class DiffusionExpert:
    def __init__(self, pretrained_module_path):
        self.module = EDMModule.load_from_checkpoint(pretrained_module_path)

    def expert_prob(self, action, observation):
        with torch.no_grad():
            logp = self.module.diffusion.logpx(x_0=action.unsqueeze(0), s=observation.unsqueeze(0))[0]
        return logp

    def expert_energy(self, t, action, observation):
        with torch.no_grad():
            energy = self.module.diffusion.p_energy(x_0=action.unsqueeze(0), t=t, s=observation.unsqueeze(0))
        return energy
