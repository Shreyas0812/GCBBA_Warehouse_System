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
            task_indices = list(range(self.nt))  # Use task indices 0..nt-1
            self._run_sga(agent_indices, task_indices, agent_paths)
        else:
            remaining_task_indices = set(range(self.nt))  # Use task indices
            # Multiple connected components: run SGA independently on each component
            # Each component gets access to all tasks (generous to baseline)
            for component in components:
                agent_indices = sorted(list(component))
                task_indices = list(remaining_task_indices)
                
                assigned_in_component = self._run_sga(agent_indices, task_indices, agent_paths)
                remaining_task_indices -= assigned_in_component  # Update remaining tasks for next components
        
        assignment = []
        for i in range(self.na):
            assignment.append(list(agent_paths[i]))

        bid_sum, makespan = self._compute_scores(agent_paths)
        
        self.assig_history.append(assignment)

        return assignment, np.round(bid_sum, 6), makespan  # Return assignment, total score, and makespan
    
    def _run_sga(self, agent_indices, task_indices, agent_paths):
        """
        Core SGA loop: sequentially assign (agent, task) pairs with highest
        marginal gain until no positive bids remain or capacity is reached.
        
        Args:
            agent_indices: List of agent indices to consider
            task_indices: List of task indices (0..nt-1) to allocate
            agent_paths: List of current agent paths (modified in place)
        
        Returns:
            Set of task indices that were assigned
        """
        available_tasks = set(task_indices)
        assigned_tasks = set()

        Nmin = min(len(available_tasks), self.Lt * len(agent_indices))

        for _ in range(Nmin):
            if not available_tasks:
                break

            best_bid = -float('inf')
            best_agent = None
            best_task_idx = None
            best_insert_pos = None

            # Evaluate all (agent, task) pairs to find the best bid
            for i in agent_indices:
                if len(agent_paths[i]) >= self.Lt:
                    continue  # Skip if agent has reached capacity

                for task_idx in available_tasks:
                    task = self.tasks[task_idx]  # task_idx is 0..nt-1
                    bid, insert_pos = self._compute_marginal_gain(i, agent_paths[i], task)

                    if bid > best_bid or (bid == best_bid and best_agent is not None and i < best_agent):
                        best_bid = bid
                        best_agent = i
                        best_task_idx = task_idx
                        best_insert_pos = insert_pos   
            
            # No more feasible assignments
            if best_agent is None:
                break
                
            # Assign the best task to the best agent
            # Store the actual task ID in the agent's path
            task_id = self.tasks[best_task_idx].id
            agent_paths[best_agent].insert(best_insert_pos, task_id)
            available_tasks.remove(best_task_idx)
            assigned_tasks.add(best_task_idx)

        return assigned_tasks
    
    def _compute_marginal_gain(self, agent_idx, agent_path, task):
        """
        Compute marginal gain of assigning a task to an agent i.e inserting a task at the optimal position in the agent's path.

        c_ij(p_i) = S_i(p_i ⊕_opt j) - S_i(p_i)

        For RPT, S_i is negative total completion time, so marginal gain is
        the best (least negative) score after insertion minus current score.
        
        Args:
            agent_idx: Index of the agent
            agent_path: Current path of the agent (list of task IDs)
            task: Task object to potentially insert
        
        Returns:
            Tuple of (marginal_gain, best_position)
        """
        # Get current path score
        current_score = self._evaluate_path(agent_idx, agent_path)
        
        best_score = -float('inf')
        best_pos = 0

        for pos in range(len(agent_path) + 1):
            # Simulate inserting the task at position pos in the agent's path
            # and compute the resulting score S_i(p_i ⊕_opt j)
            candidate_path = list(agent_path)
            candidate_path.insert(pos, task.id)  # Insert task ID at position pos
            score = self._evaluate_path(agent_idx, candidate_path)

            if score > best_score:
                best_score = score
                best_pos = pos

        # Marginal gain is the change in score
        marginal_gain = best_score - current_score
        
        return marginal_gain, best_pos

    def _evaluate_path(self, agent_idx, path):
        """
        Evaluate the score of a given path for an agent based on the chosen metric.
        For RPT, this would be negative total completion time.

        This is a placeholder function and should be implemented based on the specific metric and task characteristics.
        """
        cur_pos = self.agent_pos[agent_idx]
        cur_pos_grid = self.agent_pos_grid[agent_idx]
        speed = self.agent_speed[agent_idx]
        score = 0
        travel_time = 0

        for task_id in path:
            task = self._get_task_by_id(task_id)
            # Compute travel time from current position to task's induct station
            travel_time += self._get_distance(cur_pos, cur_pos_grid, task.induct_pos, task.induct_grid) / speed
            
            # Induct to eject time (task duration)
            travel_time += self._get_distance(task.induct_pos, task.induct_grid, task.eject_pos, task.eject_grid) / speed
            
            score -= travel_time  # RPT metric is negative total completion time
            
            travel_time = 0 # Reset travel time for next task (RPT)

            cur_pos = task.eject_pos
            cur_pos_grid = task.eject_grid

        return score
    
    def _get_task_by_id(self, task_id):
        """Look up task object by its ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise ValueError(f"Task ID {task_id} not found")
    
    def _get_distance(self, pos, pos_grid, target_pos, target_grid):
        """
        Get distance between two positions using BFS lookup (mirrors GCBBA_Agent._get_distance).
        """
        if self.grid_map is None or pos_grid is None or target_grid is None:
            return np.linalg.norm(np.array(pos) - np.array(target_pos))

        # BFS from target (target is station)
        table = self.grid_map.bfs_distances_from_station.get(target_grid)
        if table is not None and pos_grid in table:
            return table[pos_grid]

        # BFS from pos (pos is station)
        table = self.grid_map.bfs_distances_from_station.get(pos_grid)
        if table is not None and target_grid in table:
            return table[target_grid]

        # Fallback to Euclidean
        return np.linalg.norm(np.array(pos) - np.array(target_pos))
    
    def _compute_scores(self, agent_paths):
        """
        Compute total score (sum of agent scores) and makespan (max completion time across agents) for the current assignment.
        """
        total_score = 0
        makespan = 0

        for i in range(self.na):
            if not agent_paths[i]:
                continue  # Skip agents with no assigned tasks

            path_score = self._evaluate_path(i, agent_paths[i])
            
            # For total score and makespan, we use completion time (negative of RPT score)
            completion_time = -path_score
            total_score += completion_time
            
            if completion_time > makespan:
                makespan = completion_time

        return total_score, makespan

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

