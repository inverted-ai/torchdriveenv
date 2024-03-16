"""
An example showing how to define an OpenAI gym environment based on TorchDriveSim.
It uses the IAI API to provide behaviors for other vehicles and requires an access key to run.
"""
import os
import logging
import math
import inspect
import pickle
import random
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict

import gymnasium as gym
import torch
from torch import Tensor
from invertedai.common import TrafficLightState, AgentState, Point, AgentAttributes, RecurrentState

import torchdrivesim
from torchdrivesim.behavior.iai import iai_location_info_from_local, get_static_actors, IAIWrapper, \
    iai_conditional_initialize
from torchdrivesim.goals import WaypointGoal
from torchdrivesim.kinematic import KinematicBicycle
from torchdrivesim.mesh import BirdviewMesh
from torchdrivesim.rendering import renderer_from_config
from torchdrivesim.rendering.base import RendererConfig
from torchdrivesim.utils import Resolution, save_video
from torchdrivesim.utils import set_seeds
from torchdrivesim.lanelet2 import find_lanelet_directions, load_lanelet_map
from torchdrivesim.traffic_controls import TrafficLightControl, StopSignControl, YieldControl
from torchdrivesim.simulator import TorchDriveConfig, SimulatorInterface, \
    BirdviewRecordingWrapper, Simulator, HomogeneousWrapper, CollisionMetric

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaselineAlgorithm(Enum):
    """
    Method used to calculate collisions between agents.
    """
    sac = 'sac'
    ppo = 'ppo'
    a2c = 'a2c'
    td3 = 'td3'

@dataclass
class EnvConfig:
    ego_only: bool = False
    use_mock_lights: bool = True
    max_environment_steps: int = 200
    use_background_traffic: bool = True
    terminated_at_infraction: bool = True
    seed: Optional[int] = None
    simulator: TorchDriveConfig = TorchDriveConfig(renderer=RendererConfig(left_handed_coordinates=True,
                                                                           highlight_ego_vehicle=True),
                                                   collision_metric=CollisionMetric.nograd,
                                                   left_handed_coordinates=True)
    render_mode: Optional[str] = "rgb_array"
    video_filename: Optional[str] = "rendered_video.mp4"
    video_res: Optional[int] = 1024
    video_fov: Optional[float] = 200


@dataclass
class RlTrainingConfig:
    algorithm: BaselineAlgorithm = None
    parallel_env_num: Optional[int] = 2
    env: EnvConfig = EnvConfig()


@dataclass
class Scenario:
    agent_states: List[List[float]] = None
    agent_attributes: List[List[float]] = None
    recurrent_states: List[List[float]] = None


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

    def __init__(self, cfg: EnvConfig, simulator: SimulatorInterface):
        if cfg.render_mode is not None and cfg.render_mode not in self.metadata["render_modes"]:
            raise NotImplementedError
        self.render_mode = cfg.render_mode

        acceleration_range = (-1.0, 1.0)
        steering_range = (-0.3, 0.3)
        action_range = np.ndarray(shape=(2, 2), dtype=np.float32)
        action_range[:, 0] = acceleration_range
        action_range[:, 1] = steering_range
        self.max_environment_steps = cfg.max_environment_steps
        self.environment_steps = 0
        self.action_space = gym.spaces.Box(
            low=action_range[0],
            high=action_range[1],
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)

        self.reward_range = (- float('inf'), float('inf'))
        self.collision_threshold = 0.0
        self.offroad_threshold = 0.0

        self.config = cfg
        self.simulator = simulator
        self.current_action = None

        self.last_birdview = None

    # TODO: use the seed
    # TODO: return the reset info
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.simulator = self.start_sim.copy()
        self.environment_steps = 0
        self.last_birdview = None
        return self.get_obs(), {}

    def step(self, action: Tensor):
        self.environment_steps += 1
        self.simulator.step(action)
        self.last_action = self.current_action if self.current_action is not None else action
        self.current_action = action
        return self.get_obs(), self.get_reward(), self.is_terminated(), self.is_truncated(), self.get_info()

    def get_obs(self):
        birdview = self.simulator.render_egocentric().cpu().numpy()
        return birdview

    def get_reward(self):
        x = self.simulator.get_state()[..., 0]
        r = torch.zeros_like(x)
        return r

    def is_done(self):
         return self.is_truncated() or self.is_terminated()

    def is_truncated(self):
        return self.environment_steps >= self.max_environment_steps

    def is_terminated(self):
        return False

    def get_info(self):
        self.info = dict(
            offroad=self.simulator.compute_offroad(),
            collision=self.simulator.compute_collision(),
            traffic_light_violation=self.simulator.compute_traffic_lights_violations(),
            is_success=(self.environment_steps >= self.max_environment_steps),
        )
        return self.info

    def seed(self, seed=None):
        pass

    def render(self):
        if self.render_mode == 'rgb_array':
            birdview = self.simulator.render_egocentric().cpu().numpy()
            return np.transpose(birdview.squeeze(), axes=(1, 2, 0))
        else:
            raise NotImplementedError

    def mock_step(self):
        obs = np.zeros((1, 3, 64, 64)) # self.last_obs
        reward = 0
        terminated = False
        truncated = True
        info = dict(
            offroad=torch.Tensor([[0]]),
            collision=torch.Tensor([[0]]),
            traffic_light_violation=torch.Tensor([[0]]),
            is_success=False,
        )
        return obs, reward, terminated, truncated, info

    def close(self):
        if isinstance(self.simulator, BirdviewRecordingWrapper):
            bvs = self.simulator.get_birdviews()
            if len(bvs) > 1:
                save_video(bvs, self.config.video_filename)


def build_simulator(cfg: EnvConfig, location, ego_state, scenario=None, car_sequences=None, waypointseq=None):
    with torch.no_grad():
        device = torch.device("cuda")
        driving_surface_mesh_path = os.path.join(
            os.path.dirname(os.path.realpath(
                __file__)), f"{torchdrivesim.__path__[0]}/resources/maps/carla/meshes/{location}_driving_surface_mesh.pkl"
        )
        driving_surface_mesh = BirdviewMesh.unpickle(
            driving_surface_mesh_path).to(device)
        simulator_cfg = cfg.simulator
        iai_location = f'carla:{":".join(location.split("_"))}'

        if cfg.use_mock_lights:
            static_actors = get_static_actors(iai_location_info_from_local(iai_location))
            traffic_light_ids = []
            traffic_light_poses = []
            stop_sign_ids = []
            stop_sign_poses = []
            yield_sign_ids = []
            yield_sign_poses = []
            for id in static_actors:
                if static_actors[id]['agent_type'] == "traffic-light":
                    traffic_light_ids.append(id)
                    traffic_light_poses.append(static_actors[id]['pos'])
                if static_actors[id]['agent_type'] == "stop-sign":
                    stop_sign_ids.append(id)
                    stop_sign_poses.append(static_actors[id]['pos'])
                if static_actors[id]['agent_type'] == "yield":
                    yield_sign_ids.append(id)
                    yield_sign_poses.append(static_actors[id]['pos'])

            if len(traffic_light_ids) > 0:
                traffic_light_control = TrafficLightControl(location=location, pos=torch.stack(traffic_light_poses).unsqueeze(0), use_mock_lights=True, ids=traffic_light_ids)
                traffic_light_states = dict(zip(static_actors.keys(), traffic_light_control.compute_state(0).squeeze()))
                traffic_light_state_history = [{k:TrafficLightState(traffic_light_control.allowed_states[int(traffic_light_states[k])]) for k in traffic_light_states}]
            else:
                traffic_light_control = None
                traffic_light_state_history = None

            if len(stop_sign_ids) > 0:
                stop_sign_control = StopSignControl(pos=torch.stack(stop_sign_poses).unsqueeze(0), ids=stop_sign_ids)
            else:
                stop_sign_control = None

            if len(yield_sign_ids) > 0:
                yield_control = YieldControl(pos=torch.stack(yield_sign_poses).unsqueeze(0), ids=yield_sign_ids)
            else:
                yield_control = None

        else:
            traffic_light_control = None
            traffic_light_state_history = None
            stop_sign_control = None
            yield_control = None

        if cfg.ego_only:
            agent_states = torch.Tensor([ego_state[0], ego_state[1], ego_state[2], ego_state[3]]).unsqueeze(0)
            length = np.random.random() * (5.5 - 4.8) + 4.8
            width = np.random.random() * (2.2 - 1.8) + 1.8
            rear_axis_offset = np.random.random() * (0.97 - 0.82) + 0.82
            agent_attributes = torch.Tensor([length, width, rear_axis_offset]).unsqueeze(0)
            recurrent_states = torch.Tensor([0] * 132).unsqueeze(0)


        if cfg.use_background_traffic:
            background_traffic_dir = os.path.join(
                os.path.dirname(os.path.realpath(
                    __file__)), f"resources/background_traffic")
            while True:
                background_traffic_file = os.path.join(background_traffic_dir, random.choice(list(filter(lambda x: x.split("_")[0]==location, os.listdir(background_traffic_dir)))))
                with open(background_traffic_file, "rb") as f:
                    background_traffic = pickle.load(f)
                if len(background_traffic["agent_states"]) + background_traffic["agent_density"] < 100:
                    break

            remain_agent_states = [AgentState(center=Point(x=ego_state[0], y=ego_state[1]), orientation=ego_state[2], speed=ego_state[3])]
            remain_agent_attributes = [background_traffic["agent_attributes"][0]]
            remain_recurrent_states = [background_traffic["recurrent_states"][0]]
            if scenario is not None:
                for agent_state in scenario.agent_states:
                    remain_agent_states.append(AgentState(center=Point(x=agent_state[0], y=agent_state[1]), orientation=agent_state[2], speed=agent_state[3]))
                for agent_attribute in scenario.agent_attributes:
                    remain_agent_attributes.append(AgentAttributes(length=agent_attribute[0], width=agent_attribute[1], rear_axis_offset=agent_attribute[2]))
                for recurrent_state in scenario.recurrent_states:
                    remain_recurrent_states.append(RecurrentState(packed=recurrent_state))

            for i in range(len(background_traffic["agent_states"])):
                agent_state = background_traffic["agent_states"][i]
                if math.dist(ego_state[:2], (agent_state.center.x, agent_state.center.y)) > 100:
                    remain_agent_states.append(agent_state)
                    remain_agent_attributes.append(background_traffic["agent_attributes"][i])
                    remain_recurrent_states.append(background_traffic["recurrent_states"][i])
            agent_attributes, agent_states, recurrent_states = iai_conditional_initialize(location=iai_location,
                   agent_count=max(95 - len(remain_agent_states), background_traffic["agent_density"]), agent_attributes=remain_agent_attributes, agent_states=remain_agent_states, recurrent_states=remain_recurrent_states,
                   center=tuple(ego_state[:2]), traffic_light_state_history=traffic_light_state_history)


        agent_attributes, agent_states = agent_attributes.unsqueeze(
            0), agent_states.unsqueeze(0)
        agent_attributes, agent_states = agent_attributes.to(device).to(torch.float32), agent_states.to(device).to(
            torch.float32)
        kinematic_model = KinematicBicycle()
        kinematic_model.set_params(lr=agent_attributes[..., 2])
        kinematic_model.set_state(agent_states)
        renderer = renderer_from_config(
            simulator_cfg.renderer, static_mesh=driving_surface_mesh)

        traffic_controls = {}
        if traffic_light_control is not None:
            traffic_controls["traffic_light"] = traffic_light_control
        if stop_sign_control is not None:
            traffic_controls["stop_sign"] = stop_sign_control
        if yield_control is not None:
            traffic_controls["yield_sign"] = yield_control

        # BxAxNxMx2
        agent_num = agent_states.shape[1]
        waypoints = dict(vehicle=torch.Tensor(waypointseq).unsqueeze(-2).unsqueeze(0).unsqueeze(0).expand(-1, agent_num, -1, -1, -1).to(device))
        # BxAxNxM
        mask = dict(vehicle=torch.tensor([False] + [True] * (len(waypointseq) - 1)).unsqueeze(-1).unsqueeze(0).unsqueeze(0).expand(-1, agent_num, -1, -1).to(device))
        waypoint_goals = WaypointGoal(waypoints, mask)

        simulator = Simulator(
            cfg=simulator_cfg, road_mesh=driving_surface_mesh,
            kinematic_model=dict(vehicle=kinematic_model), agent_size=dict(vehicle=agent_attributes[..., :2]),
            initial_present_mask=dict(vehicle=torch.ones_like(
                agent_states[..., 0], dtype=torch.bool)),
            renderer=renderer,
            traffic_controls=traffic_controls,
            waypoint_goals=waypoint_goals
        )
        simulator = HomogeneousWrapper(simulator)
        npc_mask = torch.ones(
            agent_states.shape[-2], dtype=torch.bool, device=agent_states.device)
        npc_mask[0] = False

        if not cfg.ego_only:
            simulator = IAIWrapper(
                simulator=simulator, npc_mask=npc_mask, recurrent_states=[
                    recurrent_states],
                rear_axis_offset=agent_attributes[..., 2:3], locations=[
                    iai_location],
                car_sequences=car_sequences
            )
        if cfg.render_mode == "video":
            simulator = BirdviewRecordingWrapper(
                simulator, res=Resolution(cfg.video_res, cfg.video_res), fov=cfg.video_fov, to_cpu=True)

        return simulator


class WaypointSuiteEnv(GymEnv):
    def __init__(self, cfg: EnvConfig, data: WaypointSuite):
        self.config = cfg
        set_seeds(self.config.seed, logger)
        self.locations = data.locations
        self.iai_locations = [f'carla:{":".join(location.split("_"))}' for location in self.locations]

        self.waypoint_suite = data.waypoint_suite
        self.car_sequence_suite = data.car_sequence_suite
        self.scenarios = data.scenarios
        self.lanelet_maps = {}
        for location in self.locations:
            if location not in self.lanelet_maps:
                lanelet_map_path = f"{torchdrivesim.__path__[0]}/resources/maps/carla/maps/{location}.osm"
                self.lanelet_maps[location] = load_lanelet_map(lanelet_map_path)
        super().__init__(cfg=cfg, simulator=None)

        logger.info(inspect.getsource(WaypointSuiteEnv.get_reward))

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.current_waypoint_suite_idx = np.random.randint(len(self.waypoint_suite))
        location = self.locations[self.current_waypoint_suite_idx]
        self.lanelet_map = self.lanelet_maps[location]

        self.set_start_pos()
        self.current_target_idx = 1
        self.current_target = self.waypoint_suite[self.current_waypoint_suite_idx][self.current_target_idx]

        ego_state = (self.start_point[0], self.start_point[1], self.start_orientation, self.start_speed)

        self.last_x = None
        self.last_y = None
        self.last_psi = None

        self.last_obs = None
        self.last_reward = None
        self.last_info = None

        self.reached_waypoint_num = 0
        self.environment_steps = 0

        self.simulator = build_simulator(self.config,
                                         location=location,
                                         ego_state=ego_state,
                                         scenario=self.scenarios[self.current_waypoint_suite_idx],
                                         car_sequences=self.car_sequence_suite[self.current_waypoint_suite_idx],
                                         waypointseq=self.waypoint_suite[self.current_waypoint_suite_idx])

        return self.get_obs(), {}

    def set_start_pos(self):
        self.waypoints = self.waypoint_suite[self.current_waypoint_suite_idx]
        p0 = np.array(self.waypoints[0])
        p1 = np.array(self.waypoints[1])
        # in case the start_point is offroad
        try:
            self.start_point = p0 + np.random.rand() * (p1 - p0)
            self.start_speed = np.random.rand() * 10
            self.start_orientation = float(find_lanelet_directions(lanelet_map=self.lanelet_map,
                                                                   x=self.start_point[0], y=self.start_point[1])[0]) \
                                     + np.random.normal(0, 0.1)
        except Exception as e:
            self.start_point = p0
            self.start_speed = np.random.rand() * 10
            self.start_orientation = float(find_lanelet_directions(lanelet_map=self.lanelet_map,
                                                                   x=self.start_point[0], y=self.start_point[1])[0]) \
                                     + np.random.normal(0, 0.1)

    def step(self, action: Tensor):
#        try:
        state = self.simulator.get_state()
        self.last_x = state[..., 0]
        self.last_y = state[..., 1]
        self.last_psi = state[..., 2]
        self.last_speed = state[..., 3]

        obs, reward, terminated, truncated, info = super().step(action)
        if self.check_reach_target():
            self.current_target_idx += 1
            if self.current_target_idx < len(self.waypoints):
                self.current_target = self.waypoints[self.current_target_idx]
            else:
                self.current_target = None
        self.last_obs = obs
        self.last_reward = reward
        self.last_info = info
#        except Exception as e:
#            obs, reward, terminated, truncated, info = self.mock_step()
        return obs, reward, terminated, truncated, info

    def check_reach_target(self):
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        return (self.current_target is not None) and (math.dist((x, y), self.current_target) < 3)

    def get_reward(self):
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        psi = self.simulator.get_state()[..., 2]

        d = math.dist((x, y), (self.last_x, self.last_y)) if (self.last_x is not None) and (self.last_y is not None) else 0
        distance_reward = 1 if d > 0.5 else 0
        psi_reward = (1 - math.cos(psi - self.last_psi)) * (-20.0) if (self.last_psi is not None) else 0
        if self.check_reach_target():
            reach_target_reward = 10
            self.reached_waypoint_num += 1
        else:
            reach_target_reward = 0
        r = torch.zeros_like(x)
        r += reach_target_reward + distance_reward + psi_reward
        return r

    def is_terminated(self):
        if self.config.terminated_at_infraction:
            return (self.simulator.compute_offroad() > 0) or (self.simulator.compute_collision() > 0) or ((self.simulator.compute_traffic_lights_violations()) > 0)
        else:
            return False

    def get_info(self):
        psi = self.simulator.get_state()[..., 2]
        speed = self.simulator.get_state()[..., 3]
        reached_waypoint_num = self.reached_waypoint_num
        self.info = dict(
            offroad=self.simulator.compute_offroad(),
            collision=self.simulator.compute_collision(),
            traffic_light_violation=self.simulator.compute_traffic_lights_violations(),
            is_success=(self.environment_steps >= self.max_environment_steps),
            reached_waypoint_num=reached_waypoint_num,
            psi_smoothness=((self.last_psi - psi) / 0.1).norm(p=2).item(),
            speed_smoothness=((self.last_speed - speed) / 0.1).norm(p=2).item()
        )
        return self.info


class SingleAgentWrapper(gym.Wrapper):
    """
    Removes batch and agent dimensions from the environment interface.
    Only safe if those dimensions are both singletons.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, _ = super().reset(**kwargs)
        return self.transform_out(obs), _

    def step(self, action: Tensor):
        action = torch.Tensor(action).unsqueeze(0).unsqueeze(0).to("cuda")
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self.transform_out(obs)
        reward = self.transform_out(reward)
        terminated = self.transform_out(terminated)
        info = self.transform_out(info)

        return obs, reward, terminated, truncated, info

    def transform_out(self, x):
        if torch.is_tensor(x):
            t = x.squeeze(0).squeeze(0).cpu()
        elif isinstance(x, dict):
            t = {k: self.transform_out(v) for (k, v) in x.items()}
        elif isinstance(x, np.ndarray):
            t = self.transform_out(torch.tensor(x)).cpu().numpy()
        else:
            t = x
        return t

    def transform_in(self, x):
        if torch.is_tensor(x):
            t = x.unsqueeze(0).unsqueeze(0)
        elif isinstance(x, dict):
            t = {k: self.transform_in(v) for (k, v) in x.items()}
        else:
            t = x
        return t

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()
