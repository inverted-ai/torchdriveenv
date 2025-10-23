import copy
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
from invertedai.common import AgentAttributes, AgentState, Point, RecurrentState, RECURRENT_SIZE

from torchdrivesim.behavior.iai import IAINPCController
from torchdrivesim.behavior.replay import ReplayController
from torchdrivesim.goals import WaypointGoal
from torchdrivesim.kinematic import KinematicBicycle
from torchdrivesim.lanelet2 import find_lanelet_directions
from torchdrivesim.map import find_map_config, traffic_controls_from_map_config
from torchdrivesim.rendering import renderer_from_config
from torchdrivesim.rendering.base import RendererConfig
from torchdrivesim.simulator import compute_agent_collisions_metric
from torchdrivesim.simulator import (
    CollisionMetric,
    CompoundNPCController,
    Simulator,
    TorchDriveConfig,
)
from torchdrivesim.traffic_lights import current_light_state_tensor_from_controller
from torchdrivesim.utils import Resolution

from torchdriveenv.helpers import save_video, set_seeds
from torchdriveenv.iai import iai_conditional_initialize

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    seed: Optional[int] = None
    simulator: TorchDriveConfig = field(
        default_factory=lambda: TorchDriveConfig(
            renderer=RendererConfig(
                left_handed_coordinates=True,
                highlight_ego_vehicle=True,
            ),
            collision_metric=CollisionMetric.nograd,
            # collision_metric=CollisionMetric.discs,
            left_handed_coordinates=True,
        )
    )
    render_mode: Optional[str] = "rgb_array"
    video_filename: Optional[str] = "rendered_video.mp4"
    video_res: Optional[int] = 1024
    video_fov: Optional[float] = 500
    ego_rotate: bool = False
    device: Optional[str] = None

@dataclass
class Scenario:
    agent_states: Optional[List[List[float]]] = None
    agent_attributes: Optional[List[List[float]]] = None
    recurrent_states: Optional[List[List[float]]] = None


@dataclass
class WaypointSuite:
    locations: List[str] = None
    waypoint_suite: List[List[List[float]]] = None
    car_sequence_suite: List[Optional[Dict[int, List[List[float]]]]] = None
    scenarios: List[Optional[Scenario]] = None


class GymEnv(gym.Env):

    metadata = {
        "render_modes": ["video", "rgb_array"],
        "render_fps": 10
    }

    def __init__(self, cfg: EnvConfig, simulator: Optional[Simulator]):
        if cfg.render_mode and cfg.render_mode not in self.metadata["render_modes"]:
            raise NotImplementedError(f"Unsupported render mode: {cfg.render_mode}")

        self.config = cfg
        self.render_mode = cfg.render_mode

        if cfg.device is not None:
            self.device = torch.device(cfg.device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.simulator: Optional[Simulator] = (
            simulator.to(self.device) if simulator is not None else None
        )
        self.start_sim: Optional[Simulator] = (
            self.simulator.copy() if self.simulator is not None else None
        )

        acceleration_range = (-1.0, 1.0)
        steering_range = (-0.3, 0.3)
        action_range = np.ndarray(shape=(2, 2), dtype=np.float32)
        action_range[:, 0] = acceleration_range
        action_range[:, 1] = steering_range
        self.action_space = gym.spaces.Box(
            low=action_range[0],
            high=action_range[1],
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
                "speed": gym.spaces.Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array([200.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "birdview_image": gym.spaces.Box(
                    low=0, high=255, shape=(3, 64, 64), dtype=np.uint8
                ),
                "prev_action": self.action_space,
            }
        )

        self.max_environment_steps = cfg.max_environment_steps
        self.environment_steps = 0
        self.current_action = None
        self.prev_action = torch.zeros(2, dtype=torch.float32, device=self.device)

        self.recording = False
        self.frames: List[torch.Tensor] = []
        self.render_res: Optional[Resolution] = None
        self.render_fov: Optional[float] = None
        self.ego_rotate = cfg.ego_rotate
        self.video_filename = cfg.video_filename

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        if self.start_sim is None:
            raise RuntimeError("Simulator has not been initialised.")
        self.simulator = self.start_sim.copy()
        self.simulator.to(self.device)

        self.environment_steps = 0
        self.prev_action = torch.zeros(2, dtype=torch.float32, device=self.device)
        self.last_obs = None
        if self.recording:
            self.frames.clear()

        return self.get_obs(), {}

    def step(self, action: np.array):
        if self.simulator is None:
            raise RuntimeError("Simulator has not been initialised.")

        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action_tensor.dim() == 0:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0).unsqueeze(0)
        elif action_tensor.dim() == 2:
            action_tensor = action_tensor.unsqueeze(0)

        self.environment_steps += 1
        self.simulator.step(action_tensor)
        self.prev_action = self.current_action if self.current_action is not None else action
        self.prev_action = self.prev_action.squeeze(0).squeeze(0)
        self.current_action = action

        if self.recording:
            frame = (
                self.simulator.render_egocentric(
                    ego_rotate=self.ego_rotate,
                    res=self.render_res, fov=self.render_fov
                )
                .detach()
                .cpu()
            )
            self.frames.append(frame)

        obs = self.get_obs()
        reward = self.get_reward()
        truncated = self.is_truncated()
        terminated = self.is_terminated()
        info = self.get_info()

        return obs, reward, terminated, truncated, info

    def get_obs(self):
        if self.simulator is None:
            raise RuntimeError("Simulator has not been initialised.")
        state = self.simulator.get_state()
        speed = state[0, 0, 3].unsqueeze(0).detach().clone()
        birdview = self.simulator.render_egocentric()[0, 0].detach().clone()
        prev_action = self.prev_action.detach().clone()

        obs = {
            "speed": speed,
            "birdview_image": birdview.to(torch.uint8),
            "prev_action": prev_action,
        }
        return obs

    def get_reward(self) -> float:
        # Default reward (override in subclasses)
        return 0.0

    def _ego_collision_score(self) -> float:
        state = self.simulator.get_all_agent_state().detach().cpu().numpy()  # B x All x 4
        size = self.simulator.get_all_agent_size().detach().cpu().numpy()  # B x All x 3
        present = self.simulator.get_all_agent_present_mask().detach().cpu().numpy()  # B x All (bool)

        boxes = np.concatenate([state[..., :2], size[..., :2], state[..., 2:3]], axis=-1)  # (B, All, 5)

        per_batch_boxes = [boxes[b, present[b]] for b in range(boxes.shape[0])]
        per_batch_masks = [present[b, present[b]] for b in range(present.shape[0])]
        scores = compute_agent_collisions_metric(per_batch_boxes, per_batch_masks, present)
        return float(scores[0, 0])  # ego is agent 0

    def is_truncated(self) -> bool:
        return self.environment_steps >= self.max_environment_steps

    def is_terminated(self) -> bool:
        # Default termination (override in subclasses)
        return False

    def get_info(self) -> Dict[str, float]:
        if self.simulator is None:
            raise RuntimeError("Simulator has not been initialised.")
        offroad = self.simulator.compute_offroad().detach().cpu().numpy()
        # collision = self.simulator.compute_collision().detach().cpu().numpy()
        collision = self._ego_collision_score()
        tl_violation = (
            self.simulator.compute_traffic_lights_violations().detach().cpu().numpy()
        )
        return {
            "offroad": float(offroad.squeeze()),
            "collision": float(collision.squeeze()),
            "traffic_light_violation": float(tl_violation.squeeze()),
            "is_success": self.environment_steps >= self.max_environment_steps,
        }

    def render(
        self,
        mode: str = "rgb_array",
        res: Optional[Resolution] = None,
        fov: Optional[float] = None,
        filename: Optional[str] = None,
    ):
        if self.simulator is None:
            raise RuntimeError("Simulator has not been initialised.")
        if mode == "video":
            self.recording = True
            self.render_res = res or Resolution(self.config.video_res, self.config.video_res)
            self.render_fov = fov or self.config.video_fov
            self.video_filename = filename or self.config.video_filename
            self.frames.clear()
            return None
        if mode == "rgb_array":
            frame = (
                self.simulator.render_egocentric(ego_rotate = self.ego_rotate, res=res, fov=fov)[0, 0]
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
            )
            return frame
        raise NotImplementedError(f"Unsupported render mode: {mode}")

    def close(self):
        if self.recording and self.frames and self.video_filename:
            fps = self.metadata.get("render_fps", 10)
            save_video(self.frames, self.video_filename, batch_index=0, fps=fps)
        self.recording = False
        self.frames.clear()


def build_simulator(
    cfg: EnvConfig,
    map_cfg,
    device: torch.device,
    ego_state,
    scenario: Optional[Scenario] = None,
    car_sequences: Optional[Dict[int, List[List[float]]]] = None,
    waypointseq: Optional[List[List[float]]] = None,
) -> Simulator:
    traffic_light_controller = map_cfg.traffic_light_controller
    if traffic_light_controller is not None:
        traffic_light_controller = copy.deepcopy(traffic_light_controller)
    traffic_light_ids = [stopline.actor_id for stopline in map_cfg.stoplines if stopline.agent_type == "traffic_light"]
    driving_surface_mesh = map_cfg.road_mesh.to(device)

    traffic_controls = traffic_controls_from_map_config(map_cfg)
    for control in traffic_controls.values():
        control.to(device)
    if "traffic_light" in traffic_controls and traffic_light_controller is not None:
        light_state = current_light_state_tensor_from_controller(
            traffic_light_controller, traffic_light_ids).unsqueeze(0)
        traffic_controls["traffic_light"].set_state(light_state)

    if cfg.ego_only:
        agent_states = torch.tensor(ego_state, dtype=torch.float32).unsqueeze(0)
        length = np.random.random() * (5.5 - 4.8) + 4.8
        width = np.random.random() * (2.2 - 1.8) + 1.8
        rear_axis_offset = np.random.random() * (0.97 - 0.82) + 0.82
        agent_attributes = torch.tensor([length, width, rear_axis_offset]).unsqueeze(0)
        recurrent_states: List[RecurrentState] = []
    else:
        # Refined background-traffic bootstrap: same sampling logic as original, but wrapped in _initialise_agents with validated tensors, explicit no-file failure, configurable scenario overrides, and the current controller light-state in the traffic history.
        agent_states, agent_attributes, recurrent_states = _initialise_agents(cfg, map_cfg, ego_state, scenario)

    agent_attributes = agent_attributes.to(torch.float32).unsqueeze(0).to(device)
    agent_states = agent_states.to(torch.float32).unsqueeze(0).to(device)

    ego_attributes = agent_attributes[..., :1, :]
    npc_attributes = agent_attributes[..., 1:, :]
    ego_states = agent_states[..., :1, :]
    npc_states = agent_states[..., 1:, :]

    kinematic_model = KinematicBicycle()
    kinematic_model.set_params(lr=ego_attributes[..., 2])
    kinematic_model.set_state(ego_states)
    kinematic_model.to(device)

    renderer = renderer_from_config(cfg.simulator.renderer)
    # renderer.color_map['ego'] = (255, 0, 0)  # RGB in 0–255
    # renderer.color_map['vehicle'] = (0, 128, 255)  # optional for other actors

    if waypointseq is None:
        waypointseq = [[ego_state[0], ego_state[1]]]

    # Build the ego waypoint goal in the layout WaypointGoal expects:
    #   [batch, agent, collection_idx, waypoint_idx, xy].
    # We only simulate the ego in this tensor, so the agent axis stays size 1
    # (no expand needed).  Each waypoint forms its own length-1 “collection” so that
    # when the ego reaches one, the simulator only marks that single waypoint as
    # completed; the remaining markers stay active and keep rendering in birdview.

    waypoint_tensor = (torch.tensor(waypointseq, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(-2))
    mask = (torch.tensor([False] + [True] * (len(waypointseq) - 1), dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(-1))
    waypoint_goals = WaypointGoal(waypoint_tensor, mask)

    agent_type_names = ["ego", "vehicle"]
    ego_agent_types = torch.zeros_like(ego_states[..., 0], dtype=torch.long, device=device)

    npc_controller = None
    if not cfg.ego_only and npc_states.shape[1] > 0:
        npc_present_mask = torch.ones_like(npc_states[..., 0], dtype=torch.bool, device=device)
        npc_types = torch.ones_like(npc_present_mask, dtype=torch.long, device=device)  # index 1 = "vehicle"
        ia_controller = IAINPCController(
            npc_size=npc_attributes[..., :2],
            npc_state=npc_states,
            npc_lr=npc_attributes[..., 2],
            location=map_cfg.iai_location_name,
            npc_present_mask=npc_present_mask,
            npc_types=npc_types,
            agent_type_names=agent_type_names,
            traffic_light_controller=traffic_light_controller,
        )
        ia_controller.recurrent_state = recurrent_states
        ia_controller.to(device)

        replay_controller = None
        if car_sequences:
            replay_controller = _build_replay_controller(
                npc_attributes,
                car_sequences,
                device,
            )
        if replay_controller is not None:
            controller_indices = torch.zeros_like(npc_present_mask, dtype=torch.long)
            for idx in car_sequences.keys():
                if idx < controller_indices.shape[1]:
                    controller_indices[:, idx] = 1
            npc_controller = CompoundNPCController(
                [ia_controller, replay_controller], controller_indices
            )
        else:
            npc_controller = ia_controller

    simulator = Simulator(
        cfg=cfg.simulator,
        road_mesh=driving_surface_mesh,
        kinematic_model=kinematic_model,
        agent_size=ego_attributes[..., :2],
        initial_present_mask=torch.ones_like(ego_states[..., 0], dtype=torch.bool),
        renderer=renderer,
        traffic_controls=traffic_controls,
        waypoint_goals=waypoint_goals,
        lanelet_map=[map_cfg.lanelet_map],
        npc_controller=npc_controller,
        agent_types=ego_agent_types,
        agent_type_names=agent_type_names,
    )
    simulator.to(device)
    return simulator


def _coerce_recurrent_state(raw_state):
    if isinstance(raw_state, RecurrentState):
        packed = list(raw_state.packed)
    elif isinstance(raw_state, dict):
        packed = list(raw_state.get("packed", []))
    elif isinstance(raw_state, list):
        packed = list(raw_state)
    else:
        raise TypeError(f"Unsupported recurrent state format: {type(raw_state)!r}")

    if len(packed) < RECURRENT_SIZE:
        packed = packed + [0.0] * (RECURRENT_SIZE - len(packed))
    elif len(packed) > RECURRENT_SIZE:
        packed = packed[:RECURRENT_SIZE]

    return RecurrentState(packed=packed)


def _initialise_agents(
    cfg: EnvConfig,
    map_cfg,
    ego_state,
    scenario: Optional[Scenario],
):
    if not cfg.use_background_traffic:
        raise RuntimeError("Background traffic must be enabled for non-ego runs.")

    background_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "resources",
        "background_traffic",
    )
    map_suffix = map_cfg.name.replace("carla_", "")
    background_files = [
        f
        for f in os.listdir(background_dir)
        if f.endswith(".json") and f.split("_")[1] == map_suffix
    ]
    if not background_files:
        raise RuntimeError(f"No background traffic files for map {map_suffix}.")

    while True:
        file_path = os.path.join(background_dir, random.choice(background_files))
        with open(file_path, "r", encoding="utf-8") as handle:
            background_json = json.load(handle)

        background_states = [
            AgentState.model_validate(agent_state)
            for agent_state in background_json["agent_states"]
        ]
        if len(background_states) + background_json["agent_density"] < 100:
            break

    remain_states = [
        AgentState(
            center=Point(x=ego_state[0], y=ego_state[1]),
            orientation=ego_state[2],
            speed=ego_state[3],
        )
    ]
    remain_attributes = [
        AgentAttributes.model_validate(background_json["agent_attributes"][0])
    ]
    remain_recurrent = [
        _coerce_recurrent_state(background_json["recurrent_states"][0])
    ]

    if scenario is not None:
        for agent_state in scenario.agent_states or []:
            remain_states.append(
                AgentState(
                    center=Point(x=agent_state[0], y=agent_state[1]),
                    orientation=agent_state[2],
                    speed=agent_state[3],
                )
            )
        for agent_attr in scenario.agent_attributes or []:
            remain_attributes.append(
                AgentAttributes(
                    length=agent_attr[0],
                    width=agent_attr[1],
                    rear_axis_offset=agent_attr[2],
                )
            )
        for recurrent_state in scenario.recurrent_states or []:
            remain_recurrent.append(_coerce_recurrent_state(recurrent_state))

    for idx, agent_state in enumerate(background_states):
        if (
            math.dist(
                ego_state[:2],
                (agent_state.center.x, agent_state.center.y),
            )
            > 100
        ):
            remain_states.append(agent_state)
            remain_attributes.append(
                AgentAttributes.model_validate(
                    background_json["agent_attributes"][idx]
                )
            )
            remain_recurrent.append(
                _coerce_recurrent_state(background_json["recurrent_states"][idx])
            )

    (
        agent_attributes_tensor,
        agent_states_tensor,
        recurrent_states,
    ) = iai_conditional_initialize(
        location=map_cfg.iai_location_name,
        agent_count=max(1, 95 - len(remain_states)),
        agent_attributes=remain_attributes,
        agent_states=remain_states,
        recurrent_states=remain_recurrent,
        center=tuple(ego_state[:2]),
        traffic_light_state_history=[
            map_cfg.traffic_light_controller.current_state_with_name
        ],
    )
    return agent_states_tensor, agent_attributes_tensor, recurrent_states


def _build_replay_controller(
    npc_attributes: torch.Tensor,
    car_sequences: Dict[int, List[List[float]]],
    device: torch.device,
) -> Optional[ReplayController]:
    if not car_sequences:
        return None
    npc_count = npc_attributes.shape[1]
    horizon = max(len(seq) for seq in car_sequences.values())
    npc_states = torch.zeros(
        (1, npc_count, horizon, 4), dtype=torch.float32, device=device
    )
    npc_masks = torch.zeros(
        (1, npc_count, horizon), dtype=torch.bool, device=device
    )
    for agent_idx, seq in car_sequences.items():
        if agent_idx >= npc_count:
            continue
        seq_tensor = torch.tensor(seq, dtype=torch.float32, device=device)
        length = seq_tensor.shape[0]
        npc_states[0, agent_idx, :length] = seq_tensor
        npc_masks[0, agent_idx, :length] = True

    controller = ReplayController(
        npc_size=npc_attributes[..., :2],
        npc_states=npc_states,
        npc_present_masks=npc_masks,
        agent_type_names=["vehicle"],
    )
    controller.to(device)
    return controller


class WaypointSuiteEnv(GymEnv):
    def __init__(self, cfg: EnvConfig, data: WaypointSuite):
        if cfg.device is not None:
            self.torch_device = torch.device(cfg.device)
        else:
            self.torch_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        set_seeds(cfg.seed, logger)
        self.map_cfgs = [find_map_config(f"carla_{location}") for location in data.locations]

        self.waypoint_suite = data.waypoint_suite
        self.car_sequence_suite = data.car_sequence_suite
        self.scenarios = data.scenarios
        super().__init__(cfg=cfg, simulator=None)

        self.current_waypoint_suite_idx = 0
        self.current_target_idx = 0
        self.current_target: Optional[List[float]] = None
        self.last_position: Optional[tuple] = None
        self.last_psi: Optional[float] = None
        self.last_speed: Optional[float] = None
        self.reached_waypoint_num = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.current_waypoint_suite_idx = np.random.randint(len(self.waypoint_suite))
        map_cfg = self.map_cfgs[self.current_waypoint_suite_idx]
        self.lanelet_map = map_cfg.lanelet_map

        self._set_start_pos()
        self.current_target_idx = 1
        self.current_target = self.waypoint_suite[self.current_waypoint_suite_idx][self.current_target_idx]

        ego_state = (
            self.start_point[0],
            self.start_point[1],
            self.start_orientation,
            self.start_speed,
        )
        self.simulator = build_simulator(
            cfg=self.config,
            map_cfg=map_cfg,
            device=self.torch_device,
            ego_state=ego_state,
            scenario=self.scenarios[self.current_waypoint_suite_idx],
            car_sequences=self.car_sequence_suite[self.current_waypoint_suite_idx],
            waypointseq=self.waypoint_suite[self.current_waypoint_suite_idx],
        )
        self.start_sim = self.simulator.copy()

        self.last_position = None
        self.last_psi = None
        self.last_speed = None
        self.reached_waypoint_num = 0

        return super().reset(seed=seed, options=options)

    def step(self, action):
        if self.simulator is None:
            raise RuntimeError("Simulator has not been initialised.")

        state = self.simulator.get_state()[0, 0]
        self.last_x = state[..., 0]
        self.last_y = state[..., 1]
        self.last_psi = state[..., 2]
        self.last_speed = state[..., 3]

        obs, reward, terminated, truncated, info = super().step(action)

        if self._check_reach_target():
            self.current_target_idx += 1
            if self.current_target_idx < len(
                self.waypoint_suite[self.current_waypoint_suite_idx]
            ):
                self.current_target = self.waypoint_suite[self.current_waypoint_suite_idx][
                    self.current_target_idx
                ]
            else:
                self.current_target = None
        self.last_obs = obs
        self.last_reward = reward
        self.last_info = info
        return obs, reward, terminated, truncated, info

    def _set_start_pos(self):
        waypoints = self.waypoint_suite[self.current_waypoint_suite_idx]
        p0 = np.array(waypoints[0])
        p1 = np.array(waypoints[1])
        try:
            self.start_point = p0 + np.random.rand() * (p1 - p0)
            self.start_speed = np.random.rand() * 10
            orientation = float(
                find_lanelet_directions(
                    lanelet_map=self.lanelet_map,
                    x=self.start_point[0],
                    y=self.start_point[1],
                )[0]
            )
            self.start_orientation = orientation + np.random.normal(0, 0.1)
        except Exception:
            self.start_point = p0
            self.start_speed = np.random.rand() * 10
            orientation = float(
                find_lanelet_directions(
                    lanelet_map=self.lanelet_map,
                    x=self.start_point[0],
                    y=self.start_point[1],
                )[0]
            )
            self.start_orientation = orientation + np.random.normal(0, 0.1)

    def _check_reach_target(self) -> bool:
        if self.current_target is None or self.simulator is None:
            return False
        state = self.simulator.get_state()[0, 0]
        position = (state[0].item(), state[1].item())
        return math.dist(position, self.current_target) < 3.0

    def get_reward(self):
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        psi = self.simulator.get_state()[..., 2]

        d = math.dist((x, y), (self.last_x, self.last_y)) if (self.last_x is not None) and (
                    self.last_y is not None) else 0
        distance_reward = self.config.distance_bonus if d > self.config.distance_cutoff else 0
        psi_reward = (1 - math.cos(psi - self.last_psi)) * (- self.config.heading_penalty) if (
                    self.last_psi is not None) else 0
        if self._check_reach_target():
            reach_target_reward = self.config.waypoint_bonus
            self.reached_waypoint_num += 1
        else:
            reach_target_reward = 0
        r = torch.zeros_like(x)
        r += reach_target_reward + distance_reward + psi_reward
        return r.item()

    def is_terminated(self) -> bool:
        if not self.config.terminated_at_infraction or self.simulator is None:
            return False
        offroad = self.simulator.compute_offroad().item()
        # collision = self.simulator.compute_collision().item()
        collision = self._ego_collision_score()
        tl_violation = self.simulator.compute_traffic_lights_violations().item()
        return (offroad > 0) or (collision > 0) or (tl_violation > 0)

    def get_info(self) -> Dict[str, float]:
        if self.simulator is None:
            return {}
        state = self.simulator.get_state()[0, 0]
        position = (state[0].item(), state[1].item())
        psi = state[2].item()
        speed = state[3].item()
        distance = (
            math.dist(position, self.last_position)
            if self.last_position is not None
            else 0.0
        )
        offroad = self.simulator.compute_offroad().item()
        # collision = self.simulator.compute_collision().item()
        collision = self._ego_collision_score()
        tl_violation = self.simulator.compute_traffic_lights_violations().item()

        info = {
            "offroad": offroad,
            "collision": collision,
            "traffic_light_violation": tl_violation,
            "is_success": self.environment_steps >= self.max_environment_steps,
            "reached_waypoint_num": self.reached_waypoint_num,
            "psi_smoothness": 0.0,
            "psi_reward": 0.0,
            "dist_reward": self.config.distance_bonus
            if distance > self.config.distance_cutoff
            else 0.0,
            "speed_smoothness": 0.0,
        }
        if self.last_psi is not None:
            info["psi_smoothness"] = ((self.last_psi - psi) / 0.1).norm(p=2).item()
            info["psi_reward"] = (
                1 - math.cos(psi - self.last_psi)
            ) * -self.config.heading_penalty
        if self.last_speed is not None:
            info["speed_smoothness"] = ((self.last_speed - speed) / 0.1).norm(p=2).item()
        return info


class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._transform_out(obs), info

    def step(self, action):
        device = getattr(self.env, "device", torch.device("cpu"))
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=device)
        if action_tensor.dim() == 0:
            action_tensor = action_tensor.unsqueeze(0)
        action_tensor = action_tensor.unsqueeze(0).unsqueeze(0)
        obs, reward, terminated, truncated, info = super().step(action_tensor)
        return (
            self._transform_out(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._transform_out(info),
        )

    def _transform_out(self, value):
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
            if arr.ndim == 0:
                arr = arr.reshape(1)
            return arr
        if isinstance(value, dict):
            return {k: self._transform_out(v) for k, v in value.items()}
        return value

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()
