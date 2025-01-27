import os
import logging
import math
import json
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import gymnasium as gym
from invertedai.common import AgentState, Point, AgentAttributes, RecurrentState

from torchdrivesim.behavior.iai import IAIWrapper, iai_drive
from torchdrivesim.goals import WaypointGoal
from torchdrivesim.kinematic import BicycleNoReversing, KinematicBicycle
from torchdrivesim.rendering import renderer_from_config
from torchdrivesim.utils import Resolution
from torchdrivesim.lanelet2 import find_lanelet_directions
from torchdrivesim.map import find_map_config, traffic_controls_from_map_config
from torchdrivesim.traffic_lights import current_light_state_tensor_from_controller
from torchdrivesim.simulator import SimulatorInterface, \
    BirdviewRecordingWrapper, Simulator, HomogeneousWrapper

from torchdriveenv.common import WaypointSuite
from torchdriveenv.configs import RealisticMetric, EnvConfig
from torchdriveenv.helpers import save_video, set_seeds, sample_waypoints_from_graph
from torchdriveenv.iai import iai_conditional_initialize
from torchdriveenv.diffusion_expert import DiffusionExpert
from torchdriveenv.offline_critic import OfflineCritic

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



@dataclass
class Node:
    id: int
    point: Tuple[float]
    next_node_ids: List[int]
    next_edges: List[float]


@dataclass
class StepData:
    obs_birdview: List
    ego_action: Tuple
    ego_state: List
    recurrent_states: List[List[float]]
    reward: float
    info: Dict
    waypoint: Tuple
#    q: float


@dataclass
class EpisodeData:
    location: str
    step_data: List[StepData]


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
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)

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

    def step(self, action: np.array):
        self.environment_steps += 1
        self.simulator.step(action)
        self.last_action = self.current_action if self.current_action is not None else action
        self.current_action = action
        return self.get_obs(), self.get_reward(), self.is_terminated(), self.is_truncated(), self.get_info()

    def get_obs(self):
        birdview = self.simulator.render_egocentric().cpu().numpy().astype(np.uint8)
        return birdview

    def get_reward(self):
        x = self.simulator.get_state()[..., 0]
        r = np.zeros(x.shape)
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
        obs = np.zeros((1, 3, 64, 64))  # self.last_obs
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
                print("Saving video...")
                print(self.config.video_filename)
                save_video(bvs, self.config.video_filename)


def build_simulator(cfg: EnvConfig, map_cfg, device, ego_state, scenario=None, car_sequences=None, waypointseq=None):
    with torch.no_grad():
        traffic_light_controller = map_cfg.traffic_light_controller
        initial_light_state_name = traffic_light_controller.current_state_with_name
        traffic_light_ids = [
            stopline.actor_id for stopline in map_cfg.stoplines if stopline.agent_type == 'traffic_light']
        driving_surface_mesh = map_cfg.road_mesh

        traffic_controls = traffic_controls_from_map_config(map_cfg)
        traffic_controls = {
            key: traffic_controls[key] for key in traffic_controls}
        traffic_controls['traffic_light'].set_state(current_light_state_tensor_from_controller(
            traffic_light_controller, traffic_light_ids).unsqueeze(0))

        if cfg.ego_only:
            agent_states = torch.Tensor(
                [ego_state[0], ego_state[1], ego_state[2], ego_state[3]]).unsqueeze(0)
            length = np.random.random() * (5.5 - 4.8) + 4.8
            width = np.random.random() * (2.2 - 1.8) + 1.8
            rear_axis_offset = np.random.random() * (0.97 - 0.82) + 0.82
            agent_attributes = torch.Tensor(
                [length, width, rear_axis_offset]).unsqueeze(0)
            recurrent_states = torch.Tensor([0] * 132).unsqueeze(0)
        else:
            if cfg.use_background_traffic:
                background_traffic_dir = os.path.join(
                    os.path.dirname(os.path.realpath(
                        __file__)), f"resources/background_traffic")
                while True:
                    background_traffic_file = os.path.join(background_traffic_dir, random.choice(list(filter(
                        lambda x: x.split("_")[1] == map_cfg.name[6:], os.listdir(background_traffic_dir)))))
                    with open(background_traffic_file, "r") as f:
                        background_traffic_json = json.load(f)
                    background_traffic = {}
                    background_traffic['location'] = background_traffic_json['location']
                    background_traffic['agent_density'] = background_traffic_json['agent_density']
                    background_traffic['random_seed'] = background_traffic_json['random_seed']
                    background_traffic['agent_states'] = [AgentState.model_validate(
                        agent_state) for agent_state in background_traffic_json['agent_states']]
                    background_traffic['agent_attributes'] = [AgentAttributes.model_validate(
                        agent_attribute) for agent_attribute in background_traffic_json['agent_attributes']]
                    background_traffic['recurrent_states'] = [RecurrentState.model_validate(
                        recurrent_state) for recurrent_state in background_traffic_json['recurrent_states']]

                    if len(background_traffic["agent_states"]) + background_traffic["agent_density"] < 100:
                        break

                remain_agent_states = [AgentState(center=Point(
                    x=ego_state[0], y=ego_state[1]), orientation=ego_state[2], speed=ego_state[3])]
                remain_agent_attributes = [
                    background_traffic["agent_attributes"][0]]
                remain_recurrent_states = [
                    background_traffic["recurrent_states"][0]]
                if scenario is not None:
                    for agent_state in scenario.agent_states:
                        remain_agent_states.append(AgentState(center=Point(
                            x=agent_state[0], y=agent_state[1]), orientation=agent_state[2], speed=agent_state[3]))
                    for agent_attribute in scenario.agent_attributes:
                        remain_agent_attributes.append(AgentAttributes(
                            length=agent_attribute[0], width=agent_attribute[1], rear_axis_offset=agent_attribute[2]))
                    for recurrent_state in scenario.recurrent_states:
                        remain_recurrent_states.append(
                            background_traffic["recurrent_states"][0])

                for i in range(len(background_traffic["agent_states"])):
                    agent_state = background_traffic["agent_states"][i]
                    if math.dist(ego_state[:2], (agent_state.center.x, agent_state.center.y)) > 100:
                        remain_agent_states.append(agent_state)
                        remain_agent_attributes.append(
                            background_traffic["agent_attributes"][i])
                        remain_recurrent_states.append(
                            background_traffic["recurrent_states"][i])
                agent_attributes, agent_states, recurrent_states = iai_conditional_initialize(location=map_cfg.iai_location_name,
                                                                                              agent_count=max(65 - len(remain_agent_states), background_traffic["agent_density"]), agent_attributes=remain_agent_attributes, agent_states=remain_agent_states, recurrent_states=remain_recurrent_states,
                                                                                              center=tuple(ego_state[:2]), traffic_light_state_history=[initial_light_state_name])

        agent_attributes, agent_states = agent_attributes.unsqueeze(
            0), agent_states.unsqueeze(0)
        agent_attributes, agent_states = agent_attributes.to(torch.float32), agent_states.to(
            torch.float32)
        kinematic_model = BicycleNoReversing()
        kinematic_model.set_params(lr=agent_attributes[..., 2])
        kinematic_model.set_state(agent_states)
        renderer = renderer_from_config(
            cfg.simulator.renderer, static_mesh=driving_surface_mesh)

        # BxAxNxMx2
        agent_num = agent_states.shape[1]
        waypoints = dict(vehicle=torch.Tensor(
            waypointseq).unsqueeze(-2).unsqueeze(0).unsqueeze(0).expand(-1, agent_num, -1, -1, -1))
        # BxAxNxM
        mask = dict(vehicle=torch.tensor([False] + [True] * (len(waypointseq) - 1)
                                         ).unsqueeze(-1).unsqueeze(0).unsqueeze(0).expand(-1, agent_num, -1, -1))
        waypoint_goals = WaypointGoal(waypoints, mask)

        simulator = Simulator(
            cfg=cfg.simulator, road_mesh=driving_surface_mesh,
            kinematic_model=dict(vehicle=kinematic_model), agent_size=dict(vehicle=agent_attributes[..., :2]),
            initial_present_mask=dict(vehicle=torch.ones_like(
                agent_states[..., 0], dtype=torch.bool)),
            renderer=renderer,
            traffic_controls=traffic_controls,
            waypoint_goals=waypoint_goals
        )
#            waypoint_goals=None
#            waypoint_goals=waypoint_goals
        simulator = HomogeneousWrapper(simulator)
        npc_mask = torch.ones(
            agent_states.shape[-2], dtype=torch.bool, device=agent_states.device)
        npc_mask[0] = False

        if not cfg.ego_only:

            if car_sequences is not None and len(car_sequences) > 0:
                T = max([len(car_seq) for car_seq in car_sequences.values()])
                replay_states = torch.zeros((1, agent_num, T, 4))
                replay_mask = torch.zeros(
                    replay_states.shape[:3], dtype=torch.bool)
                replay_states[:, list(car_sequences.keys()), :, :] = torch.Tensor(
                    list(car_sequences.values())).unsqueeze(0)
                replay_mask[:, list(car_sequences.keys()), :] = True
            else:
                replay_states = None
                replay_mask = None

            simulator = IAIWrapper(
                simulator=simulator, npc_mask=npc_mask, recurrent_states=[
                    recurrent_states],
                rear_axis_offset=agent_attributes[..., 2:3], locations=[
                    map_cfg.iai_location_name],
                traffic_light_controller=traffic_light_controller,
                traffic_light_ids=traffic_light_ids,
                replay_states=replay_states,
                replay_mask=replay_mask
            )
        if cfg.render_mode == "video":
            simulator = BirdviewRecordingWrapper(
                simulator, res=Resolution(cfg.video_res, cfg.video_res), fov=cfg.video_fov, to_cpu=True)
        simulator.to(device)

        return simulator


class WaypointSuiteEnv(GymEnv):
    def __init__(self, cfg: EnvConfig, data: WaypointSuite):
        self.config = cfg
        if cfg.device is None:
            self.torch_device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.torch_device = torch.device(cfg.device)

        set_seeds(self.config.seed, logger)
        self.map_cfgs = [find_map_config(
            f"carla_{location}") for location in data.locations]

        self.waypoint_suite = data.waypoint_suite
        self.waypoint_graphs = data.waypoint_graphs
        self.car_sequence_suite = data.car_sequence_suite
        self.scenarios = data.scenarios
#        if self.config.use_expert_similarity:
        if self.config.realistic_metric == RealisticMetric.expert_mse:
            self.expert_kinematic_model = KinematicBicycle(left_handed=True)
        elif self.config.realistic_metric == RealisticMetric.expert_diffusion_elbo:
            self.diffusion_expert = DiffusionExpert(self.config.pretrained_diffusion_expert_path)
        if self.config.pretrained_offline_critic_path is not None:
            self.offline_critic = OfflineCritic(self.config.pretrained_offline_critic_path)
        super().__init__(cfg=cfg, simulator=None)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.current_waypoint_suite_idx = np.random.randint(
            len(self.waypoint_suite))
        map_cfg = self.map_cfgs[self.current_waypoint_suite_idx]
        self.lanelet_map = map_cfg.lanelet_map
        if (self.waypoint_graphs is not None) and (self.waypoint_graphs[self.current_waypoint_suite_idx] is not None):
            self.waypoint_suite[self.current_waypoint_suite_idx] = sample_waypoints_from_graph(self.waypoint_graphs[self.current_waypoint_suite_idx])

        self.set_start_pos()
        self.current_target_idx = 1
        self.current_target = self.waypoint_suite[self.current_waypoint_suite_idx][self.current_target_idx]

        ego_state = (self.start_point[0], self.start_point[1],
                     self.start_orientation, self.start_speed)

        self.last_x = None
        self.last_y = None
        self.last_psi = None

        self.last_obs = None
        self.last_reward = None
        self.last_info = None

        self.reached_waypoint_num = 0
        self.environment_steps = 0

        self.simulator = build_simulator(self.config,
                                         map_cfg=map_cfg,
                                         ego_state=ego_state,
                                         scenario=self.scenarios[self.current_waypoint_suite_idx],
                                         car_sequences=self.car_sequence_suite[self.current_waypoint_suite_idx],
                                         waypointseq=self.waypoint_suite[self.current_waypoint_suite_idx],
                                         device=self.torch_device)
        self.last_obs = self.get_obs()
        self.obs_list = [self.last_obs, self.last_obs, self.last_obs]
        return self.last_obs, {}

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

    def expert_prediction(self):
#        location = f'carla:{":".join(self.map_cfgs[self.current_waypoint_suite_idx].name.split("_"))}'
        location = self.map_cfgs[self.current_waypoint_suite_idx].iai_location_name
        agent_states = self.simulator.get_innermost_simulator().get_state()[
            "vehicle"].squeeze().cpu().numpy()
        agent_attributes = self.simulator._agent_attributes
        recurrent_states = self.simulator._recurrent_states
        traffic_lights_states = self.simulator._traffic_light_controller.current_state_with_name
        waypoint_for_ego = self.current_target

        obs = {"location": location,
               "agent_states": agent_states,
               "agent_attributes": agent_attributes[0],
               "recurrent_states": recurrent_states[0],
               "traffic_lights_states": traffic_lights_states,
               "waypoint_for_ego": waypoint_for_ego}

        states, recurrent_states = iai_drive(location=obs["location"],
                                             agent_states=obs["agent_states"],
                                             agent_attributes=obs["agent_attributes"],
                                             recurrent_states=obs["recurrent_states"],
                                             traffic_lights_states=obs["traffic_lights_states"],
                                             waypoint_for_ego=obs["waypoint_for_ego"])

        action = self.expert_kinematic_model.fit_action(
            future_state=states[0], current_state=torch.Tensor(obs["agent_states"][0])).to(self.torch_device)
        return action

    def step(self, action: np.array):
        #        try:
#        if self.config.use_expert_similarity:
        if self.config.realistic_metric == RealisticMetric.expert_mse:
            self.expert_action = self.expert_prediction()
        self.action = action
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
        self.obs_list.append(obs)
#        except Exception as e:
#            obs, reward, terminated, truncated, info = self.mock_step()
        return obs, reward, terminated, truncated, info

    def check_reach_target(self):
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        return (self.current_target is not None) and (math.dist((x, y), self.current_target) < 3)

    def get_reward(self):
#        if self.config.use_expert_similarity:
        encounter_infractions = ((self.simulator.compute_offroad() > 0) or (
            self.simulator.compute_collision() > 0) or (self.simulator.compute_traffic_lights_violations() > 0)).item()

        safety_reward = -self.config.infraction_penalty if encounter_infractions else 0

        if self.config.realistic_metric == RealisticMetric.expert_mse:
            expert_action = self.expert_action.clip(min=torch.Tensor([-1.0, -0.3]).to(self.torch_device), max=torch.Tensor([1.0, 0.3]).to(self.torch_device))
            realistic_reward = torch.log(1 - torch.linalg.vector_norm(expert_action - self.action.squeeze()) / torch.linalg.vector_norm(torch.Tensor([2.0, 0.6]).to(self.torch_device)))
        elif self.config.realistic_metric == RealisticMetric.expert_diffusion_elbo:
            stacked_obs = np.concatenate(self.obs_list[-3:], axis=-3)
            action = self.action.squeeze()
            action[1] *= 10
#            realistic_reward = min(self.diffusion_expert.expert_prob(action=action, observation=torch.Tensor(stacked_obs).to(self.torch_device).squeeze()), 0)
#            realistic_reward = self.diffusion_expert.expert_logp_from_energy(action=action, observation=torch.Tensor(stacked_obs).to(self.torch_device).squeeze())
            realistic_reward = 0
        else:
#            if encounter_infractions:
#                r = float('-inf')
#            else:
#                expert_action = self.expert_action.clip(min=torch.Tensor([-1.0, -0.3]).to(self.torch_device), max=torch.Tensor([1.0, 0.3]).to(self.torch_device))
#                r = torch.log(1 - torch.linalg.vector_norm(expert_action - self.action.squeeze()) / torch.linalg.vector_norm(torch.Tensor([2.0, 0.6]).to(self.torch_device)))
##                print("expert_action: ", expert_action)
##                print("action: ", self.action)
##            print("reward: ", r)
#            return r

            x = self.simulator.get_state()[..., 0]
            y = self.simulator.get_state()[..., 1]
            psi = self.simulator.get_state()[..., 2]

            d = math.dist((x, y), (self.last_x, self.last_y)) if (
                self.last_x is not None) and (self.last_y is not None) else 0
            distance_reward = self.config.distance_bonus if d > self.config.distance_cutoff else 0
            psi_reward = (1 - math.cos(psi - self.last_psi)) * \
                (- self.config.heading_penalty) if (self.last_psi is not None) else 0
            if self.check_reach_target():
                reach_target_reward = self.config.waypoint_bonus
                self.reached_waypoint_num += 1
            else:
                reach_target_reward = 0
    #        r += reach_target_reward + distance_reward + psi_reward
            realistic_reward = reach_target_reward + distance_reward + psi_reward

#        r = torch.zeros_like(x)
#        print("safety_reward")
#        print(safety_reward)
#        print("realistic_reward")
#        print(realistic_reward)
        r = safety_reward + realistic_reward
#        r = r.item()
#        encounter_infractions = ((self.simulator.compute_offroad() > 0) or (
#            self.simulator.compute_collision() > 0) or (self.simulator.compute_traffic_lights_violations() > 0)).item()
#        if encounter_infractions:
#            r = float('-inf')
#        else:
#            #            r = 0
#            r /= 1000

#            r = -1000.0
        return r

    def is_terminated(self):
        if self.config.terminated_at_infraction:
            return ((self.simulator.compute_offroad() > 0) or (self.simulator.compute_collision() > 0) or ((self.simulator.compute_traffic_lights_violations()) > 0)).item()
        else:
            return False

    def get_info(self):
        x = self.simulator.get_state()[..., 0]
        y = self.simulator.get_state()[..., 1]
        psi = self.simulator.get_state()[..., 2]
        speed = self.simulator.get_state()[..., 3]
        d = math.dist((x, y), (self.last_x, self.last_y)) if (
            self.last_x is not None) and (self.last_y is not None) else 0
        reached_waypoint_num = self.reached_waypoint_num
        self.info = dict(
            offroad=self.simulator.compute_offroad(),
            collision=self.simulator.compute_collision(),
            traffic_light_violation=self.simulator.compute_traffic_lights_violations(),
            reached_waypoint_num=reached_waypoint_num,
            is_success=((self.environment_steps >= self.max_environment_steps) and (self.current_target_idx >= 3)),
            psi_smoothness=((self.last_psi - psi) / 0.1).norm(p=2).item(),
            psi_reward=(1 - math.cos(psi - self.last_psi)) *
            (- self.config.heading_penalty),
            dist_reward=self.config.distance_bonus if d > self.config.distance_cutoff else 0,
            speed_smoothness=((self.last_speed - speed) / 0.1).norm(p=2).item()
        )
        return self.info
#            is_success=(self.current_target_idx >= 3),


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

    def step(self, action: np.array):
        action = torch.Tensor(action).unsqueeze(
            0).unsqueeze(0).to(self.torch_device)
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
