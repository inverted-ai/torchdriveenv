from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from torchdrivesim.rendering.base import RendererConfig
from torchdrivesim.simulator import TorchDriveConfig, CollisionMetric



class RealisticMetric(Enum):
    manual = 'manual'
    expert_mse = 'expert_mse'
    expert_diffusion_elbo = 'expert_diffusion_elbo'


@dataclass
class EnvConfig:
    ego_only: bool = False
    max_environment_steps: int = 200
    frame_stack: int = 3
    waypoint_bonus: float = 100.
    heading_penalty: float = 25.
    distance_bonus: float = 1.
    distance_cutoff: float = 0.5
    use_background_traffic: bool = True
    terminated_at_infraction: bool = True
    infraction_penalty: float = float('inf')
#    use_expert_similarity: bool = False
    realistic_metric: RealisticMetric = RealisticMetric.expert_diffusion_elbo
    pretrained_diffusion_expert_path: Optional[str] = "pretrained_edm_module/model.ckpt"
#    pretrained_offline_critic_path: Optional[str] = "pretrained_offline_critic_module/model.ckpt"
    pretrained_offline_critic_path: Optional[str] = None
    seed: Optional[int] = None
    simulator: TorchDriveConfig = TorchDriveConfig(renderer=RendererConfig(left_handed_coordinates=True,
                                                                           highlight_ego_vehicle=True),
                                                   collision_metric=CollisionMetric.nograd,
                                                   left_handed_coordinates=True)
    render_mode: Optional[str] = "rgb_array"
    video_filename: Optional[str] = "rendered_video.mp4"
    video_res: Optional[int] = 1024
    video_fov: Optional[float] = 500
    device: Optional[str] = None
