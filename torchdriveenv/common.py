from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


@dataclass
class Scenario:
    agent_states: List[List[float]] = None
    agent_attributes: List[List[float]] = None
    recurrent_states: List[List[float]] = None


@dataclass
class Node:
    id: int
    point: Tuple[float]
    next_node_ids: List[int]
    next_edges: List[float]


@dataclass
class WaypointSuite:
    locations: List[str] = None
    waypoint_suite: List[List[List[float]]] = None
    car_sequence_suite: List[Optional[Dict[int, List[List[float]]]]] = None
    scenarios: List[Optional[Scenario]] = None
    waypoint_graphs: List[List[Node]] = None
