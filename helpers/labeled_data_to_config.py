import sys
sys.path.insert(0, "torchdrivesim")

import os
import json
import random
from omegaconf import OmegaConf

from torchdrivesim.gym_env import Scenario
from torchdrivesim.env_utils import load_waypoint_suite_env_config


json_dir = "labeled_data"
json_files = os.listdir(json_dir)

waypoint_suite_env_config = load_waypoint_suite_env_config("env_configs/iai_env.yml")

waypoint_suite_env_config.locations = []
waypoint_suite_env_config.waypointsuite = []
waypoint_suite_env_config.scenarios = []
waypoint_suite_env_config.car_sequence_suite = []
waypoint_suite_env_config.traffic_light_state_suite = []
waypoint_suite_env_config.stop_sign_suite = []
MAX_ENVIRONMENT_STEP = 200


for json_file in json_files:
    if json_file[-5:] != ".json":
        continue
    location = json_file.split('_')[0]
    waypoint_suite_env_config.locations.append(location)
    json_path=os.path.join(json_dir, json_file)
    with open(json_path) as f:
        data = json.load(f)

    waypoints = []
    for state in data['individual_suggestions']['0']['states']:
        waypoint = [state['center']['x'], state['center']['y']]
        waypoints.append(waypoint)
    waypoint_suite_env_config.waypointsuite.append(waypoints)

    scenario = None
    car_sequences = None

    if ("predetermined_agents" in data) and (data["predetermined_agents"] is not None):
        agent_states = []
        agent_attributes = []
        recurrent_states = []
        for id in data["predetermined_agents"]:
            agent = data["predetermined_agents"][id]
            if len(agent['states']) == 1:
                speed = random.randint(5, 10)
            else:
                speed = 0
            agent_states.append([agent['states']['0']['center']['x'], agent['states']['0']['center']['y'],
                                 agent['states']['0']['orientation'], speed])
            agent_attributes.append([agent['static_attributes']['length'],
                                     agent['static_attributes']['width'],
                                     agent['static_attributes']['rear_axis_offset']])
            recurrent_states.append([0] * 132)
        if len(agent_states) > 0:
            scenario = Scenario(agent_states=agent_states,
                                agent_attributes=agent_attributes,
                                recurrent_states=recurrent_states)

        car_sequences = {}
        for id in data["predetermined_agents"]:
            agent = data["predetermined_agents"][id]
            if len(agent['states']) > 1:
                car_sequences[int(id)] = []
                speed = 0
                for i in agent['states']:
                    car_sequences[int(id)].append([agent['states'][i]['center']['x'], agent['states'][i]['center']['y'], 
                                                   agent['states'][i]['orientation'], speed])

    waypoint_suite_env_config.scenarios.append(scenario)
    waypoint_suite_env_config.car_sequence_suite.append(car_sequences)

    waypoint_suite_env_config.traffic_light_state_suite.append(None)
    waypoint_suite_env_config.stop_sign_suite.append(None)


env_config_path = f"env_configs/fixed_waypoint_suite_env_config.yml"
OmegaConf.save(config=waypoint_suite_env_config, f=env_config_path)
