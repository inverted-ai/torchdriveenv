import torch
from edm.pl_modules import EDMModule

class DiffusionExpert:
    def __init__(self, model_path):
        self.model = EDMModule.load_from_checkpoint(model_path)

    def expert_prob(self, action, observation):
        with torch.no_grad():
            logp = self.model.diffusion.logpx(x_0=action.unsqueeze(0), s=observation.unsqueeze(0))[0]
        return logp
