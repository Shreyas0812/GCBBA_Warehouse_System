"""
SGA (Sequential Greedy Algorithm) Orchestrator for warehouse task allocation.

Centralized upper bound baseline that GCBBA provably converges to.
Uses the same RPT bidding metric and BFS distance lookups as GCBBA.

When the communication graph is disconnected, SGA runs independently
within each connected component (generous to baseline).

Interface matches GCBBA_Orchestrator for drop-in replacement.
"""

import time
import numpy as np
import networkx as nx

from gcbba.GCBBA_Task import GCBBA_Task

class SGA_Orchestrator:
    """SGA Orchestrator class for centralized sequential greedy task allocation"""

    def __init__(self, G, D, char_t, char_a, Lt=1, metric="RPT", task_ids=None, grid_map=None):
        self.G = G
        # int, number of agents
        self.na = G.shape[0]
        # int, number of tasks
        self.nt = len(char_t)
        # capacity per agent
        self.Lt = Lt
        # task characteristics
        self.char_t = char_t
        # agent characteristics
        self.char_a = char_a
        # original task IDs — if None, defaults to 0..nt-1 (backward compatible)
        self.task_ids = task_ids if task_ids is not None else list(range(self.nt))
        # list of all agents
        self.agents = []
        # list of all tasks
        self.tasks = []
        
        # clock launch
        self.start_time = time.perf_counter()
        
        self.metric = metric
        self.D = D
        self.grid_map = grid_map
        
        # Initialize Tasks
        for j in range(self.nt):
            self.tasks.append(GCBBA_Task(id=self.task_ids[j], char_t=self.char_t[j], grid_map=self.grid_map))

        # Initialize Agents
        self.agent_pos = []
        self.agent_pos_grid = []
        self.agent_speed = []
        for i in range(self.na):
            pos = np.array(self.char_a[i][:3])  # Extract x, y, z coordinates
            speed = self.char_a[i][3]  # Extract speed
            self.agent_pos.append(pos)
            self.agent_speed.append(speed)
            if self.grid_map is not None:
                self.agent_pos_grid.append(self.grid_map.continuous_to_grid(*pos))
            else:
                self.agent_pos_grid.append(None)
        
        # Tracking 
        self.assig_history = []
        self.bid_history = []
        self.max_times = []

    def launch_agents(self, method=None, detector=None):
        """
        Running SGA allocation
        """
        G_nx = nx.from_numpy_array(self.G)
        components = list(nx.connected_components(G_nx))

        agent_paths = [[] for _ in range(self.na)]

        if len(components) == 1:
            # Single connected component: run SGA on all agents and tasks
            agent_indices = list(range(self.na))
            task_indices = list(self.task_ids)
            self._run_sga(agent_indices, task_indices, agent_paths)
        else:
            # Multiple connected components: run SGA independently on each component
            # Each component gets access to all tasks (generous to baseline)
            for component in components:
                agent_indices = sorted(list(component))
                task_indices = list(self.task_ids)
                self._run_sga(agent_indices, task_indices, agent_paths)

        assignment = []
        for i in range(self.na):
            assignment.append(list(agent_paths[i]))
        
        self.assig_history.append(assignment)

        return assignment, None, None  # Placeholder for assignment, total score, and makespan
    
    def _run_sga(self, agent_indices, task_indices, agent_paths):
        """
        Core SGA loop: sequentially assign (agent, task) pairs with highest
        marginal gain until no positive bids remain or capacity is reached.
        """
        print(f"Running SGA on agents {agent_indices} and tasks {task_indices}")


if __name__ == "__main__":
    import yaml
    import os
    from math import ceil
    import sys
    
    # Add parent directory to path to import tools_warehouse
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, '..')
    sys.path.insert(0, parent_dir)
    
    from gcbba.tools_warehouse import create_graph_with_range, agent_init, task_init
    
    def read_warehouse_config(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    seed = 5  # Random seed for reproducibility
    np.random.seed(seed)

    # Get the path to the config file
    config_path = os.path.join(current_dir, '..', 'config', 'gridworld_warehouse_small.yaml')
    
    # Read the configuration
    config = read_warehouse_config(config_path)
    
    # Access the parameters
    params = config['create_gridworld_node']['ros__parameters']
    
    # SGA specific parameters
    na = len(params['agent_positions']) // 3  # Number of agents
    tasks_per_induct_station = 10  # Number of tasks per induct station
    xlim = [0, int(params['grid_width']) * params['grid_resolution']]
    ylim = [0, int(params['grid_height']) * params['grid_resolution']]
    sp_lim = [1, 5]  # Speed limits in units/sec
    dur_lim = [1, 10]  # Task duration limits in seconds
    comm_range = 30  # Communication range in units
    
    # Extract agent positions from config (reshape flattened array)
    agent_pos_flat = params['agent_positions']
    agent_positions = [(agent_pos_flat[i], agent_pos_flat[i+1], agent_pos_flat[i+2], i//3 + 1) 
                       for i in range(0, len(agent_pos_flat), 3)]
    
    # Extract induct station positions from config
    induct_pos_flat = params['induct_stations']
    induct_positions = [(induct_pos_flat[i], induct_pos_flat[i+1], induct_pos_flat[i+2], induct_pos_flat[i+3]) 
                        for i in range(0, len(induct_pos_flat), 4)]
    
    # Extract eject station positions from config
    eject_pos_flat = params['eject_stations']
    eject_positions = [(eject_pos_flat[i], eject_pos_flat[i+1], eject_pos_flat[i+2], eject_pos_flat[i+3]) 
                       for i in range(0, len(eject_pos_flat), 4)]
    
    nt = len(induct_positions) * tasks_per_induct_station  # Number of tasks 
    Lt = ceil(nt / na)  # Tasks per agent

    # Create communication graph based on distance (agent-to-agent only)
    raw_graph, G = create_graph_with_range(agent_positions, comm_range)
    
    D = nx.diameter(raw_graph)
    
    # Initialize agents and tasks from warehouse config
    agents = agent_init(agent_positions, sp_lim=sp_lim)
    tasks = task_init(induct_positions, eject_positions, task_per_induct_station=tasks_per_induct_station)

    # Initialize SGA orchestrator
    orch_sga = SGA_Orchestrator(G, D, tasks, agents, Lt)

    t0 = time.time()
    assig, tot_score, makespan = orch_sga.launch_agents()
    tf0 = np.round(1000 * (time.time() - t0))

    print("SGA - total score = {}; max score = {}; time = {} ms".format(tot_score, makespan, tf0))
    print("Assignment = {}".format(assig))

