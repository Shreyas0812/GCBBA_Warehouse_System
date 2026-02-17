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
