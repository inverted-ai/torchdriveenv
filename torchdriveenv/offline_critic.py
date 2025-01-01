import torch
# from offline_critic.pl_modules import CriticModule

class OfflineCritic:
    def __init__(self, pretrained_module_path):
        print("state_dict")
        print(torch.load(pretrained_module_path).keys())

#        self.module = CriticModule.load_from_checkpoint(model_path)

    def q1_critic(self, action, observation):
        with torch.no_grad():
            energy = self.module.critic_model.q1_forward(observation, action)
        return energy
