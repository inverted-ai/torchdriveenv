import os
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
