import json
import os
import random
from omegaconf import OmegaConf

import torchdriveenv
from torchdriveenv.gym_env import EnvConfig, Scenario, WaypointSuite


def construct_env_config(raw_config):
    env_config = EnvConfig(**raw_config)
    return env_config


def load_env_config(yaml_path):
    config_from_yaml = OmegaConf.to_object(OmegaConf.load(yaml_path))
    return construct_env_config(config_from_yaml)


def load_waypoint_suite_data(yaml_path):
    data_from_yaml = OmegaConf.to_object(OmegaConf.load(yaml_path))
    waypoint_suite_data = WaypointSuite(**data_from_yaml)
    if waypoint_suite_data.scenarios is not None:
        waypoint_suite_data.scenarios = [Scenario(agent_states=scenario["agent_states"],
                                                  agent_attributes=scenario["agent_attributes"],
                                                  recurrent_states=scenario["recurrent_states"])
                                         if scenario is not None else None for scenario in waypoint_suite_data.scenarios]
    return waypoint_suite_data


def load_labeled_data(data_dir):
    json_files = os.listdir(data_dir)

    waypoint_suite_env_config = WaypointSuite()

    waypoint_suite_env_config.locations = []
    waypoint_suite_env_config.waypoint_suite = []
    waypoint_suite_env_config.scenarios = []
    waypoint_suite_env_config.car_sequence_suite = []
    waypoint_suite_env_config.traffic_light_state_suite = []
    waypoint_suite_env_config.stop_sign_suite = []


    for json_file in json_files:
        if json_file[-5:] != ".json":
            continue
        location = json_file.split('_')[1]
        waypoint_suite_env_config.locations.append(location)
        json_path=os.path.join(data_dir, json_file)
        with open(json_path) as f:
            data = json.load(f)

        waypoints = []
        for state in data['individual_suggestions']['0']['states']:
            waypoint = [state['center']['x'], state['center']['y']]
            waypoints.append(waypoint)
        waypoint_suite_env_config.waypoint_suite.append(waypoints)

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
                if ("max_speed" in agent["static_attributes"]) and (agent["static_attributes"]["max_speed"] == 0):
                    car_sequences[int(id)] = []
                    speed = 0
                    for i in range(200):
                        car_sequences[int(id)].append([agent['states']['0']['center']['x'], agent['states']['0']['center']['y'],
                                                       agent['states']['0']['orientation'], speed])

                elif len(agent['states']) > 1:
                    car_sequences[int(id)] = []
                    speed = 0
                    for i in agent['states']:
                        car_sequences[int(id)].append([agent['states'][i]['center']['x'], agent['states'][i]['center']['y'],
                                                       agent['states'][i]['orientation'], speed])

        waypoint_suite_env_config.scenarios.append(scenario)
        waypoint_suite_env_config.car_sequence_suite.append(car_sequences)

        waypoint_suite_env_config.traffic_light_state_suite.append(None)
        waypoint_suite_env_config.stop_sign_suite.append(None)
    return waypoint_suite_env_config


def _load_default_data(file_name):
    for root in torchdriveenv._data_path:
        file_path = os.path.join(root, file_name)
        if os.path.exists(file_path):
            break
    else:
        return None
    return load_waypoint_suite_data(file_path)


def load_default_validation_data():
    return _load_default_data(file_name="validation_cases.yml")


def load_default_train_data():
    return _load_default_data(file_name="training_cases.yml")
