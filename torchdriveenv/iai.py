import math
import random
import torch


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
