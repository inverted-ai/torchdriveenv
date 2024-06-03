import math
import random
import torch

from typing import Tuple, List


def iai_blame(location, colliding_agents, agent_state_history, agent_attributes, traffic_light_state_history=None):
#    print("iai_blame")
#    print(colliding_agents)
    print("fake blame")
    if random.random() < 0.5:
        return [0]
    else:
        return []
#    ego_agent_speed = agent_state_history[-1][..., 0, 3]
#    other_agent_speed = agent_state_history[-1][..., colliding_agents[1], 3]
#    if ego_agent_speed > other_agent_speed:
#        return [0]
    import invertedai
    from invertedai.common import AgentState, AgentAttributes, Point
    agent_attributes = [AgentAttributes(length=at[0], width=at[1], rear_axis_offset=at[2]) for at in agent_attributes.squeeze()]
    agent_state_history = [[AgentState(center=Point(x=st[0], y=st[1]), orientation=st[2], speed=st[3]) for st in agent_states.squeeze()] for agent_states in agent_state_history[-50:]]

    response = invertedai.api.blame(location=location, colliding_agents=colliding_agents, agent_state_history=agent_state_history, agent_attributes=agent_attributes, get_confidence_score=True)
    print("agents at fault: ", response.agents_at_fault)
    print("confidence score: ", response.confidence_score)
    agents_at_fault = response.agents_at_fault
    if (response.confidence_score < 0.8) or (len(agents_at_fault) == 0):
        agents_at_fault = [0]
    return agents_at_fault


def iai_conditional_initialize(location, agent_count, agent_attributes=None, agent_states=None, recurrent_states=None, center=(0, 0), traffic_light_state_history=None):
    import invertedai

    INITIALIZE_FOV = 120
    conditional_agent_attributes = []
    conditional_agent_states = []
    conditional_recurrent_states = []
    outside_agent_states = []
    outside_agent_attributes = []
    outside_recurrent_states = []

    for i in range(len(agent_states)):
        agent_state = agent_states[i]
        dist = math.dist(center, (agent_state.center.x, agent_state.center.y))
        if dist < INITIALIZE_FOV:
            conditional_agent_states.append(agent_state)
            conditional_agent_attributes.append(agent_attributes[i])
            conditional_recurrent_states.append(recurrent_states[i])
        else:
            outside_agent_states.append(agent_state)
            outside_agent_attributes.append(agent_attributes[i])
            outside_recurrent_states.append(recurrent_states[i])

    agent_count -= len(conditional_agent_states)
#    print("agent_count: ", agent_count)
    agent_count = 0
    if agent_count > 0:
        try:
            seed = random.randint(1, 10000)
            response = invertedai.api.initialize(
                location=location,
                agent_attributes=conditional_agent_attributes,
                states_history=[conditional_agent_states],
                agent_count=agent_count,
                location_of_interest=center,
                traffic_light_state_history=traffic_light_state_history,
                random_seed = seed
            )
            agent_attribute_list = response.agent_attributes + outside_agent_attributes
            agent_state_list = response.agent_states + outside_agent_states
            recurrent_state_list = response.recurrent_states + outside_recurrent_states
        except invertedai.error.InvalidRequestError:
            agent_attribute_list = agent_attributes
            agent_state_list = agent_states
            recurrent_state_list = recurrent_states
    else:
        agent_attribute_list = agent_attributes
        agent_state_list = agent_states
        recurrent_state_list = recurrent_states

    agent_attributes = torch.stack(
        [torch.tensor(at.tolist()[:3]) for at in agent_attribute_list], dim=-2
    )
    agent_states = torch.stack(
        [torch.tensor(st.tolist()) for st in agent_state_list], dim=-2
    )
    return agent_attributes, agent_states, recurrent_state_list
