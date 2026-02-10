import os
from typing import Optional, Tuple, List
import yaml
import networkx as nx
import numpy as np

from collision_avoidance.grid_map import GridMap
from collision_avoidance.time_based_collision_avoidance import TimeBasedCollisionAvoidance

from gcbba.GCBBA_Orchestrator import GCBBA_Orchestrator
from gcbba.tools_warehouse import agent_init, create_graph_with_range, task_init

from integration.agent_state import AgentState

class IntegrationOrchestrator:
    """
    Main Integration Orchestrator

    - Run GCBBA to get task assignments
    - Assignments are sent to AgentState for execution
    - Collision Avoidance called for Path Planning and Replanning
    - Step simulation forward and update AgentState with new positions and task statuses
    - Trigger GCBBA replanning at specified intervals or when certain conditions are met (e.g. task completion, new tasks added, etc.)
    """
    
    def __init__(self, 
                 config_path: str, 
                 tasks_per_induct_station: int = 10,
                 comm_range: float = 30,
                 sp_lim: Tuple[float, float] = (1.0, 5.0),
                 rerun_interval: int = 10,
                 stuck_threshold: int = 5,
                 prediction_horizon: int = 5,
                 max_plan_time: int = 400,
                 Lt: Optional[int] = None
                 ) -> None:
        
        self.tasks_per_induct_station = tasks_per_induct_station
        self.comm_range = comm_range
        self.sp_lim = sp_lim
        self.rerun_interval = rerun_interval
        self.stuck_threshold = stuck_threshold
        self.prediction_horizon = prediction_horizon
        self.max_plan_time = max_plan_time
        self.Lt = Lt

        self.grid_map = GridMap(config_path)
        self.ca = TimeBasedCollisionAvoidance(self.grid_map)

        agent_positions, induct_positions, eject_positions = self._load_config(config_path)
        self._init_gcbba(agent_positions, induct_positions, eject_positions)
        self._init_agent_states()

    def _load_config(self, config_path: str) -> Tuple[List, List, List]:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        params = config["create_gridworld_node"]["ros__parameters"]

        agent_pos_flat = params['agent_positions']
        agent_positions = [(agent_pos_flat[i], agent_pos_flat[i+1], agent_pos_flat[i+2], i//3 + 1) 
                                for i in range(0, len(agent_pos_flat), 3)]
        
        induct_pos_flat = params['induct_stations']
        induct_positions = [(induct_pos_flat[i], induct_pos_flat[i+1], induct_pos_flat[i+2], induct_pos_flat[i+3]) 
                                 for i in range(0, len(induct_pos_flat), 4)]
        
        eject_pos_flat = params['eject_stations']
        eject_positions = [(eject_pos_flat[i], eject_pos_flat[i+1], eject_pos_flat[i+2], eject_pos_flat[i+3]) 
                                for i in range(0, len(eject_pos_flat), 4)]

        return agent_positions, induct_positions, eject_positions

    def _init_gcbba(self, agent_positions: List, induct_positions: List, eject_positions: List) -> None:
        raw_graph, G = create_graph_with_range(agent_positions, self.comm_range)
        if raw_graph.number_of_nodes() == 0:
            D = 1
        else:
            if nx.is_connected(raw_graph):
                D = nx.diameter(raw_graph)
            else:
                # If the graph is not fully connected, we can take the maximum diameter of the connected components as an approximation
                D = max(nx.diameter(raw_graph.subgraph(c)) for c in nx.connected_components(raw_graph))

        agents = agent_init(agent_positions, sp_lim=self.sp_lim)
        tasks = task_init(induct_positions, eject_positions, task_per_induct_station=self.tasks_per_induct_station)

        if self.Lt is None:
            nt = len(induct_positions) * self.tasks_per_induct_station
            na = len(agent_positions)
            Lt = int(np.ceil(nt / na))
        
        self.gcbba_orchestrator = GCBBA_Orchestrator(G, D, tasks, agents, Lt)

    def _init_agent_states(self) -> None:
        self.agent_states: List[AgentState] = []
        for idx, gcbba_agent in enumerate(self.gcbba_orchestrator.agents):
            grid_pos = self.grid_map.continuous_to_grid(float(gcbba_agent.pos[0]), float(gcbba_agent.pos[1]), float(gcbba_agent.pos[2]))
            self.agent_states.append(AgentState(agent_id=gcbba_agent.id, initial_position=grid_pos, speed=gcbba_agent.speed))

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(PROJECT_ROOT, "..", "config", "gridworld_warehouse_small.yaml")

    orchestrator = IntegrationOrchestrator(config_path)