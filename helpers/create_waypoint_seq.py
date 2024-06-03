import sys
sys.path.insert(0, "torchdrivesim")

import invertedai as iai
from invertedai.common import Point
import math
import torch
import random
from tqdm.contrib import tmap
from itertools import product
from torchdrivesim.env_utils import load_waypoint_suite_env_config
from omegaconf import OmegaConf


H_SCALE = 10
text_x_offset = 0
text_y_offset = 0.7
text_size = 7
TIMEOUT_SECS = 600
MAX_RETRIES = 10
SLACK = 2
INITIALIZE_FOV = 120
AGENT_FOV = 35
RENDERING_FOV = 1000


def get_centers(map_center, height, width, stride):
    def check_valid_center(center):
        return ((map_center[0] - width) < center[0] < (map_center[0] + width) and
                (map_center[1] - height) < center[1] < (map_center[1] + height))

    def get_neighbors(center):
        return [(center[0] + (i * stride), center[1] + (j * stride)) for i, j in list(product(*[(-1, 1), ] * 2))]

    queue, centers = [map_center], []

    while queue:
        center = queue.pop(0)
        neighbors = filter(check_valid_center, get_neighbors(center))
        queue.extend([neighbor for neighbor in neighbors if neighbor not in queue and neighbor not in centers])
        if center not in centers and check_valid_center(center):
            centers.append(center)
    return centers


def world_to_pixel(world_points, map_center, rendering_fov, res=512):
    world_points = torch.Tensor(world_points)
    map_center = torch.Tensor(map_center)
    return torch.round((world_points - map_center) / rendering_fov * res + torch.Tensor([res / 2, res / 2])).int()


def is_black_tile(tile_center, map_center, rendering_fov, birdview_image, initialize_fov=INITIALIZE_FOV):
    p0 = world_to_pixel(torch.Tensor(tile_center) - initialize_fov / 2, map_center, rendering_fov)
    p1 = world_to_pixel(torch.Tensor(tile_center) + initialize_fov / 2, map_center, rendering_fov)
    return birdview_image[p0[0]:p1[0], p0[1]:p1[1], :].mean() < 1


def area_initialization(location, agent_density, map_center, rendering_fov, birdview_image, traffic_lights_states=None, random_seed=None, 
                        width=100, height=100, stride=100, initialize_fov=INITIALIZE_FOV, get_birdview=False,
                        birdview_path=None):
    def inside_fov(center: Point, initialize_fov: float, point: Point) -> bool:
        return ((center.x - (initialize_fov / 2) < point.x < center.x + (initialize_fov / 2)) and
                (center.y - (initialize_fov / 2) < point.y < center.y + (initialize_fov / 2)))

    agent_states = []
    agent_attributes = []
    agent_rs = []
    first = True
    centers = get_centers(map_center, height, width, stride)
    # print("centers")
    # print(centers)
    for area_center in tmap(Point.fromlist, centers, total=len(centers),
                            desc=f"Initializing {location.split(':')[1]}"):

        conditional_agent = list(filter(lambda x: inside_fov(
            center=area_center, initialize_fov=initialize_fov, point=x[0].center), zip(agent_states, agent_attributes,
                                                                                       agent_rs)))
        remaining_agents = list(filter(lambda x: not inside_fov(
            center=area_center, initialize_fov=initialize_fov, point=x[0].center), zip(agent_states, agent_attributes,
                                                                                       agent_rs)))

        con_agent_state = [x[0] for x in conditional_agent]
        con_agent_attrs = [x[1] for x in conditional_agent]
        con_agent_rs = [x[2] for x in conditional_agent]
        remaining_agents_states = [x[0] for x in remaining_agents]
        remaining_agents_attrs = [x[1] for x in remaining_agents]
        remaining_agents_rs = [x[2] for x in remaining_agents]

        if len(con_agent_state) > agent_density:
            continue

        for _ in range(1):
            if is_black_tile((area_center.x, area_center.y), map_center, rendering_fov, birdview_image, initialize_fov=initialize_fov):
                print("black_tile")
                print(area_center)
                continue
            try:
                # Initialize simulation with an API cal
                response = iai.initialize(
                    location=location,
                    states_history=[con_agent_state] if len(con_agent_state) > 0 else None,
                    agent_attributes=con_agent_attrs if len(con_agent_attrs) > 0 else None,
                    agent_count=agent_density,
                    get_infractions=False,
                    traffic_light_state_history=traffic_lights_states,
                    location_of_interest=(area_center.x, area_center.y),
                    random_seed=random_seed,
                    get_birdview=get_birdview,
                )
                break
            except BaseException as e:
                print(e)
        else:
            continue
        # Filter out agents that are not inside the ROI to avoid collision with other agents not passed as conditional
        # SLACK is for removing the agents that are very close to the boundary and
        # they may collide agents not filtered as conditional
        valid_agents = list(filter(lambda x: inside_fov(
            center=area_center, initialize_fov=initialize_fov - SLACK, point=x[0].center),
            zip(response.agent_states, response.agent_attributes, response.recurrent_states)))

        valid_agent_state = [x[0] for x in valid_agents]
        valid_agent_attrs = [x[1] for x in valid_agents]
        valid_agent_rs = [x[2] for x in valid_agents]

        agent_states = remaining_agents_states + valid_agent_state
        agent_attributes = remaining_agents_attrs + valid_agent_attrs
        agent_rs = remaining_agents_rs + valid_agent_rs

        if get_birdview:
            file_path = f"{birdview_path}-{(area_center.x, area_center.y)}.jpg"
            response.birdview.decode_and_save(file_path)

    response.recurrent_states=agent_rs
    response.agent_states=agent_states
    response.agent_attributes = agent_attributes
    return response



def drive_traffic(location, agent_density, random_seed):
    location_info = iai.location_info(location=location,rendering_fov=RENDERING_FOV)
    rendered_static_map = location_info.birdview_image.decode()
    response = area_initialization(location=location,
                                   agent_density=agent_density,
                                   map_center=(location_info.map_center.x, location_info.map_center.y),
                                   rendering_fov=RENDERING_FOV,
                                   birdview_image=rendered_static_map,
                                   traffic_lights_states=None, random_seed=random_seed,
                                   width=location_info.map_fov / 2, height=location_info.map_fov / 2, stride=100,
                                   initialize_fov=INITIALIZE_FOV, get_birdview=False,
                                   birdview_path=None)

    agent_attributes = response.agent_attributes

    ego_index = random.randint(0, len(response.agent_states) - 1)
    waypoints = [[response.agent_states[ego_index].center.x, response.agent_states[ego_index].center.y]]
    last_x = None
    last_y = None
    for i in range(300):
        response = iai.drive(
                location=location,
                agent_attributes=agent_attributes,
                agent_states=response.agent_states,
                recurrent_states=response.recurrent_states,
                
        )
        x = response.agent_states[ego_index].center.x
        y = response.agent_states[ego_index].center.y
        if (math.dist((x, y), waypoints[-1]) >= 15) and (last_x is not None):
            waypoints.append([last_x, last_y])
        last_x = x
        last_y = y
        if i % 5 == 0:
            print(i)
        if len(waypoints) >= 20:
            break
    if len(waypoints) < 5:
        raise Exception("waypoints too short: ", len(waypoints))

    config.locations.append(location.split(":")[1])
    config.waypointsuite.append(waypoints)
    config.car_sequence_suite.append(None)
    config.traffic_light_state_suite.append(None)
    config.stop_sign_suite.append(None)
    config.scenarios.append(None)

    print(waypoints)


env_config_path = "env_configs/training_config.yml"
config = load_waypoint_suite_env_config(env_config_path)
config.locations = []
config.waypointsuite = []
config.car_sequence_suite = []
config.traffic_light_state_suite = []
config.stop_sign_suite = []
config.scenarios = []

# "carla:Town04", "carla:Town06",
location_list = ["carla:Town01", "carla:Town02", "carla:Town03", "carla:Town07", "carla:Town10HD"]
agent_density_list = [5, 10, 15, 20]


successful_num = 0

while successful_num < 100:
    location = random.choice(location_list)
#    agent_density = random.choice(agent_density_list)
    agent_density = 1
    random_seed = random.randint(0, 1000)
    try:
        drive_traffic(location, agent_density, random_seed)
        successful_num += 1
    except Exception as e:
        print(e)
        print('location: ', location)
        print('agent_density: ', agent_density)
        print('random_seed: ', random_seed)
    print("successful_num")
    print(successful_num)

OmegaConf.save(config=config, f=env_config_path)
