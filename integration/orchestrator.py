import os
from typing import Tuple
import yaml

from collision_avoidance.grid_map import GridMap
from collision_avoidance.time_based_collision_avoidance import TimeBasedCollisionAvoidance

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
                 ) -> None:
        self.config_path = config_path
        self.tasks_per_induct_station = tasks_per_induct_station
        self.comm_range = comm_range
        self.sp_lim = sp_lim
        self.rerun_interval = rerun_interval
        self.stuck_threshold = stuck_threshold
        self.prediction_horizon = prediction_horizon
        self.max_plan_time = max_plan_time

        self.grid_map = GridMap(config_path)
        self.ca = TimeBasedCollisionAvoidance(self.grid_map)

        self._load_config()

    def _load_config(self) -> None:
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        params = config["create_gridworld_node"]["ros__parameters"]

        agent_pos_flat = params['agent_positions']
        self.agent_positions = [(agent_pos_flat[i], agent_pos_flat[i+1], agent_pos_flat[i+2], i//3 + 1) 
                                for i in range(0, len(agent_pos_flat), 3)]
        
        induct_pos_flat = params['induct_stations']
        self.induct_positions = [(induct_pos_flat[i], induct_pos_flat[i+1], induct_pos_flat[i+2], induct_pos_flat[i+3]) 
                                 for i in range(0, len(induct_pos_flat), 4)]
        
        eject_pos_flat = params['eject_stations']
        self.eject_positions = [(eject_pos_flat[i], eject_pos_flat[i+1], eject_pos_flat[i+2], eject_pos_flat[i+3]) 
                                for i in range(0, len(eject_pos_flat), 4)]


        

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(PROJECT_ROOT, "..", "config", "gridworld_warehouse_small.yaml")
