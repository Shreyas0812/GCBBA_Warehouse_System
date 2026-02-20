"""
Integration Orchestrator for Multi-Agent Task Allocation and Path Planning
- "gcbba" (default): GCBBA with ADD bundle building + global consensus
- "sga": Centralized Sequential Greedy Algorithm
- "cbba": Standard CBBA with FULLBUNDLE + local consensus

Path Planner: Priority-based Time-Expanded A* with reservation table for collision avoidance
"""

import os
import csv
from typing import Optional, Set, Tuple, List, Dict
import yaml
import networkx as nx
import numpy as np
import time
from tqdm import tqdm
from dataclasses import dataclass

from collision_avoidance.grid_map import GridMap
from collision_avoidance.time_based_collision_avoidance import TimeBasedCollisionAvoidance

from gcbba.GCBBA_Orchestrator import GCBBA_Orchestrator
from baselines.SGA_Orchestrator import SGA_Orchestrator
from baselines.CBBA_Orchestrator import CBBA_Orchestrator

from gcbba.tools_warehouse import agent_init, create_graph_with_range, task_init

from integration.agent_state import AgentState

@dataclass
class OrchestratorEvents:
    """Events that can occur during the simulation that the orchestrator needs to handle"""
    completed_task_ids: List[int]
    stuck_agent_ids: List[int]
    gcbba_rerun: bool

class IntegrationOrchestrator:
    """
    Main Integration Orchestrator - supports GCBBA, SGA and CBBA allocation

    - Run allocation to get task assignments
    - Assignments are sent to AgentState for execution
    - Collision Avoidance called for Path Planning and Replanning
    - Step simulation forward and update AgentState with new positions and task statuses
    - Trigger GCBBA replanning at specified intervals or when certain conditions are met (e.g. task completion, new tasks added, etc.)
    """
    
    def __init__(self, 
                 config_path: str, 
                 tasks_per_induct_station: int = 10,
                 comm_range: float = 30,
                 sp_lim: Tuple[float, float] = (1.0, 1.0),
                 rerun_interval: int = 10,
                 stuck_threshold: int = 15,
                 prediction_horizon: int = 5,
                 max_plan_time: int = 400,
                 Lt: Optional[int] = None,
                 allocation_method: str = "gcbba"  # "gcbba", "sga", or "cbba"
                 ) -> None:
        
        if allocation_method not in {"gcbba", "sga", "cbba"}:
            raise ValueError(f"Invalid allocation method: {allocation_method}. Must be one of 'gcbba', 'sga', or 'cbba'.")

        self.allocation_method = allocation_method
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

        agent_positions, induct_positions, eject_positions, charging_positions = self._load_config(config_path)
        self._init_allocation(agent_positions, induct_positions, eject_positions)
        self._init_agent_states()

        # Simulation state variables
        self.current_timestep = 0
        self.last_gcbba_timestep = -self.rerun_interval  # Initialize to allow GCBBA to run at timestep 0
        self.completed_task_ids: Set[int] = set()
        self._completed_at_last_gcbba: int = 0  # Track how many tasks were completed at the time of the last GCBBA run to help determine when to trigger next run

        self.latest_assignment: List[List[int]] = []  # Store latest GCBBA assignment for reference in stepping logic

        # For Energy and Charging Logic — use dedicated charging stations from config
        self.charging_station_grid_positions = [
            self.grid_map.continuous_to_grid(float(pos[0]), float(pos[1]), float(pos[2]))
            for pos in charging_positions
        ]

    def _load_config(self, config_path: str) -> Tuple[List, List, List, List]:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        params = config["create_gridworld_node"]["ros__parameters"]

        agent_pos_flat = params['agent_positions']
        agent_positions = [(agent_pos_flat[i], agent_pos_flat[i+1], agent_pos_flat[i+2], agent_pos_flat[i+3])
                                for i in range(0, len(agent_pos_flat), 4)]

        induct_pos_flat = params['induct_stations']
        induct_positions = [(induct_pos_flat[i], induct_pos_flat[i+1], induct_pos_flat[i+2], induct_pos_flat[i+3])
                                 for i in range(0, len(induct_pos_flat), 4)]

        eject_pos_flat = params['eject_stations']
        eject_positions = [(eject_pos_flat[i], eject_pos_flat[i+1], eject_pos_flat[i+2], eject_pos_flat[i+3])
                                for i in range(0, len(eject_pos_flat), 4)]

        charging_pos_flat = params.get('charging_stations', [])
        charging_positions = [(charging_pos_flat[i], charging_pos_flat[i+1], charging_pos_flat[i+2], charging_pos_flat[i+3])
                                for i in range(0, len(charging_pos_flat), 4)]

        return agent_positions, induct_positions, eject_positions, charging_positions

    def _init_allocation(self, agent_positions: List, induct_positions: List, eject_positions: List) -> None:
        """
        Initialize the chosen allocation orchestrator (GCBBA, SGA, or CBBA) with the given configuration.
        """
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
        else:
            Lt = self.Lt

        self.all_char_t: Dict[int, np.ndarray] = {i: tasks[i] for i in range(len(tasks))}
        self.all_task_ids: Set[int] = set(self.all_char_t.keys())
        self.all_char_a = agents   # List[np.array], indexed by agent index
        self.num_agents = len(agents)

        # Initialize using GCBBA -- all methods share same agent and task initialization, so we can use the same parameters to create the orchestrator instance and then call the appropriate launch method
        self.gcbba_orchestrator_initial = GCBBA_Orchestrator(G, D, tasks, agents, Lt)

        print(f"Orchestrator initialized with {self.num_agents} agents and {len(self.all_task_ids)} tasks.")

    def _init_agent_states(self) -> None:
        self.agent_states: List[AgentState] = []
        for gcbba_agent in self.gcbba_orchestrator_initial.agents:
            grid_pos = self.grid_map.continuous_to_grid(float(gcbba_agent.pos[0]), float(gcbba_agent.pos[1]), float(gcbba_agent.pos[2]))
            self.agent_states.append(AgentState(agent_id=gcbba_agent.agent_id, initial_position=grid_pos, speed=gcbba_agent.speed, max_energy=1000))

    def run_simulation(self, timesteps: int = 100) -> None:
        pbar = tqdm(range(timesteps), desc=f"Simulation ({self.allocation_method.upper()})", leave=True)
        for _ in pbar:
            events = self.step()
            done = len(self.completed_task_ids)
            total = len(self.all_task_ids)
            pbar.set_postfix(done=f"{done}/{total}", t=self.current_timestep, refresh=False)
            # Main simulation loop logic:
            # 1. Get current task assignments from GCBBA
            # 2. Update AgentState with new assignments
            # 3. Call collision avoidance for path planning/replanning
            # 4. Step simulation forward and update AgentState with new positions and task statuses
            # 5. Trigger GCBBA replanning at specified intervals or when certain conditions are met (e.g. task completion, new tasks added, rerun time etc.)
            
            if self.completed_task_ids == self.all_task_ids:
                tqdm.write(f"All tasks completed at timestep {self.current_timestep}. Ending simulation.")
                break
    
    def step(self) -> OrchestratorEvents:
        """
        Step the simulation forward by one timestep.
        """
        if self.current_timestep == 0 or self.last_gcbba_timestep < 0:
            self.run_allocation()
            self._plan_paths()

        # Track charging state before stepping so we can detect transitions
        prev_charging_busy = {
            ast.agent_id: (ast.is_charging or ast.is_navigating_to_charger)
            for ast in self.agent_states
        }

        completed_task_ids: List[int] = []
        for agent_state in self.agent_states:
            completed = agent_state.step(self.current_timestep)

            if completed and agent_state.completed_tasks:
                completed_task_ids.append(agent_state.completed_tasks[-1].task_id)

        for task_id in completed_task_ids:
            self.completed_task_ids.add(task_id)

        # Energy and Charging Logic
        newly_available_agents = [
            ast.agent_id for ast in self.agent_states
            if prev_charging_busy.get(ast.agent_id) and not ast.is_charging and not ast.is_navigating_to_charger
        ]
        newly_charging_agents = self._check_and_start_charging()

        if newly_available_agents:
            tqdm.write(f"Agents finished charging at timestep {self.current_timestep}: {newly_available_agents}")
        if newly_charging_agents:
            tqdm.write(f"Agents starting to charge at timestep {self.current_timestep}: {newly_charging_agents}")
        if newly_charging_agents or newly_available_agents:
            self.run_allocation()

        self._plan_paths()

        # Check if we need to rerun GCBBA
        events = self._detect_events(completed_task_ids)

        if events.gcbba_rerun and self.last_gcbba_timestep != self.current_timestep:
            self.run_allocation()
            self._plan_paths()  # Replan paths immediately after GCBBA to reflect new assignments
        
        self.current_timestep += 1
        return events

    def run_allocation(self) -> None:
        """
        Run the chosen allocation method: GCBBA, SGA, or CBBA.
        """
        
        # Tasks to Exclude: completed tasks + currently executing tasks (to avoid reassigning them)
        executing_task_ids = self._get_executing_task_ids()
        excluded_task_ids = self.completed_task_ids | executing_task_ids

        # Agents to Exclude: currently charging or navigating to charger
        active_agent_indices = [
            i for i, agent_state in enumerate(self.agent_states)
            if not agent_state.is_charging and not agent_state.is_navigating_to_charger
        ]

        if len(active_agent_indices) == 0:
            tqdm.write(f"No active agents at timestep {self.current_timestep}. Skipping allocation.")
            self.last_gcbba_timestep = self.current_timestep
            return

        active_char_t = []
        active_task_ids = []

        for original_id in sorted(self.all_task_ids):
            if original_id not in excluded_task_ids:
                active_char_t.append(self.all_char_t[original_id])
                active_task_ids.append(original_id)
        
        if len(active_char_t) == 0:
            # Nothing to allocate - updating agent states with empty lists
            for agent_state in self.agent_states:
                agent_state.update_from_gcbba([], self.current_timestep)
            self.latest_assignment = [[] for _ in range(self.num_agents)]
            self.last_gcbba_timestep = self.current_timestep
            tqdm.write(f"No active tasks to allocate at timestep {self.current_timestep}. Skipping GCBBA run.")
            return
        
        # Build updated char_a only for active (non-charging) agents
        updated_char_a = []
        for i in active_agent_indices:
            agent_state = self.agent_states[i]
            if self.current_timestep > 0:
                predicted_pos = agent_state.get_predicted_position(self.prediction_horizon)
                continuous_pos = self.grid_map.grid_to_continuous(*predicted_pos)
            else:
                continuous_pos = (self.all_char_a[i][0], self.all_char_a[i][1], self.all_char_a[i][2])

            speed = float(self.all_char_a[i][3])
            agent_id_value = int(self.all_char_a[i][4])
            updated_char_a.append(np.array([continuous_pos[0], continuous_pos[1], continuous_pos[2], speed, agent_id_value]))

        # Recomputing G and D based on active agent positions only
        current_positions = []
        for i in active_agent_indices:
            agent_state = self.agent_states[i]
            pos = agent_state.get_position()
            continuous_pos = self.grid_map.grid_to_continuous(*pos)
            current_positions.append((continuous_pos[0], continuous_pos[1], continuous_pos[2], agent_state.agent_id))
        
        raw_graph, G = create_graph_with_range(current_positions, self.comm_range)
        if raw_graph.number_of_nodes() == 0:
            D = 1
        else:
            if nx.is_connected(raw_graph):
                D = nx.diameter(raw_graph)
            else:
                # If the graph is not fully connected, we can take the maximum diameter of the connected components as an approximation
                D = max(nx.diameter(raw_graph.subgraph(c)) for c in nx.connected_components(raw_graph))

        # Computing Capacity
        na = len(active_agent_indices)
        nt_active = len(active_char_t)
        if self.Lt is None:
            Lt = int(np.ceil(nt_active / na))
        else:
            Lt = self.Lt
        
        # Dispatch to the appropriate orchestrator based on the chosen allocation method
        t_allocation_start = time.perf_counter()

        if self.allocation_method == "gcbba":
            # Fresh GCBBBA Orchestrator instance with updated parameters and state
            allocator = GCBBA_Orchestrator(G, D, active_char_t, updated_char_a, Lt, task_ids=active_task_ids, grid_map=self.grid_map)
        
        elif self.allocation_method == "sga":
            allocator = SGA_Orchestrator(G, D, active_char_t, updated_char_a, Lt, task_ids=active_task_ids, grid_map=self.grid_map)
        
        elif self.allocation_method == "cbba":
            allocator = CBBA_Orchestrator(G, D, active_char_t, updated_char_a, Lt, task_ids=active_task_ids, grid_map=self.grid_map)
        
        assignment, total_score, makespan = allocator.launch_agents()

        t_allocation_end = time.perf_counter()

        allocation_time_ms = (t_allocation_end - t_allocation_start) * 1000

        tqdm.write(
            f"[t={self.current_timestep}] {self.allocation_method.upper()}: "
            f"{nt_active} active tasks, "
            f"{len(excluded_task_ids)} excluded "
            f"({len(self.completed_task_ids)} done, "
            f"{len(executing_task_ids)} executing). "
            f"Score={total_score:.2f}, Makespan={makespan:.2f}, Time={allocation_time_ms:.2f}ms"
        )
        
        self.latest_assignment = assignment

        # _build_assignment_dict uses 0-indexed positions (allocator output); remap to original agent indices
        gcbba_assignments_by_active_pos = self._build_assignment_dict(assignment)

        for active_pos, original_idx in enumerate(active_agent_indices):
            tasks_for_agent = gcbba_assignments_by_active_pos.get(active_pos, [])
            agent_state = self.agent_states[original_idx]
            agent_state.update_from_gcbba(tasks_for_agent, self.current_timestep)
            if agent_state.has_tasks() and agent_state.current_path is None:
                agent_state.needs_new_path = True

        # Charging/navigating agents get empty updates (preserves their charging state)
        inactive_indices = set(range(self.num_agents)) - set(active_agent_indices)
        for i in inactive_indices:
            self.agent_states[i].update_from_gcbba([], self.current_timestep)
        
        self.last_gcbba_timestep = self.current_timestep
        self._completed_at_last_gcbba = len(self.completed_task_ids)  # Update the count of completed tasks at the time of this GCBBA run

    def _get_executing_task_ids(self) -> Set[int]:
        executing_task_ids = set()
        for agent_state in self.agent_states:
            if agent_state.current_task is not None:
                executing_task_ids.add(agent_state.current_task.task_id)
        return executing_task_ids

    def _build_assignment_dict(self, assignment: List[List[int]]) -> Dict[int, List[int]]:
        assignments_dict: Dict[int, List[int]] = {}

        for agent_idx, task_ids in enumerate(assignment):
            tasks_for_agent: List[Dict] = []

            for task_id in task_ids:
                if task_id in self.completed_task_ids:
                    continue  # Skip already completed tasks

                char_t = self.all_char_t[task_id]

                induct_grid_pos = self.grid_map.continuous_to_grid(float(char_t[0]), float(char_t[1]), float(char_t[2]))
                eject_grid_pos = self.grid_map.continuous_to_grid(float(char_t[3]), float(char_t[4]), float(char_t[5]))

                tasks_for_agent.append({
                    "task_id": int(task_id),
                    "induct_pos": list(induct_grid_pos), # induct_grid_pos is a tuple, convert to list for easier handling in AgentState
                    "eject_pos": list(eject_grid_pos)   # eject_grid_pos
                })        
            
            assignments_dict[agent_idx] = tasks_for_agent
        
        return assignments_dict

    def _plan_paths(self) -> None:
        replan_agents = [agent_state for agent_state in self.agent_states if agent_state.needs_new_path] # Set in gcbba

        if not replan_agents:
            return

        for agent_state in replan_agents:
            self.ca.clear_agent_reservations(agent_state.agent_id)
            goal = agent_state.get_current_goal()

            if goal is None:
                start = agent_state.get_position()
                self.ca.reserve_path([start], agent_state.agent_id, start_time=self.current_timestep)
                agent_state.needs_new_path = False
                continue

            start = agent_state.get_position()

            # For charging navigation, use BFS gradient descent on the precomputed
            # distance field instead of time-expanded A*.  The goal is a fixed map
            # position so the shortest obstacle-free path is already encoded in the
            # BFS table — reconstruction is O(path_length) with no search overhead,
            # regardless of how many other agents are in the corridor.
            if agent_state.is_navigating_to_charger:
                path = self.grid_map.reconstruct_path_to_station(start, goal)
            else:
                path = self.ca.plan_path_with_reservations(
                    start=start,
                    goal=goal,
                    agent_id=agent_state.agent_id,
                    max_time=self.max_plan_time,
                    start_time=self.current_timestep
                )

            if path is None:
                path = [start]  # No path found, stay in place
            
            agent_state.assign_path(path)
            self.ca.reserve_path(path, agent_state.agent_id, start_time=self.current_timestep)

    def save_trajectories(self, path: str = "results/data/trajectories.csv") -> None:
        """Export all agent position histories to a CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["agent_id", "x", "y", "z", "timestep"])
            for agent_state in self.agent_states:
                for (x, y, z, t) in agent_state.position_history:
                    writer.writerow([agent_state.agent_id, x, y, z, t])
        print(f"Trajectories saved to {path}")

    def _detect_events(self, completed_task_ids: List[int]) -> OrchestratorEvents:

        if self.rerun_interval >= self.max_plan_time * 2:
            return OrchestratorEvents(
                completed_task_ids=completed_task_ids,
                stuck_agent_ids=[],
                gcbba_rerun=False
            )

        stuck_agent_ids: List[int] = []

        for agent_state in self.agent_states:
            if agent_state.detect_stuck(self.stuck_threshold):
                stuck_agent_ids.append(agent_state.agent_id)
                agent_state.needs_new_path = True  # Trigger replanning for stuck agents
        
        gcbba_rerun = False
        time_since_last_gcbba = self.current_timestep - self.last_gcbba_timestep

        # Cooldown to prevent excessive GCBBA reruns on close timestep events
        min_cooldown = max(3, self.rerun_interval // 3)  # At least 3 timesteps cooldown or a fraction of the rerun interval

        batch_threshold = max(2, self.num_agents // 3)  # Threshold for batch triggering based on number of agents
        completed_since_last = len(self.completed_task_ids) - self._completed_at_last_gcbba
        if completed_since_last >= batch_threshold and time_since_last_gcbba >= min_cooldown:
            gcbba_rerun = True  # Trigger GCBBA rerun if enough tasks have been completed since the last run and cooldown has passed
        
        # Handled via needs_new_plan flag in AgentState which triggers replanning (and indirectly GCBBA rerun if new paths are needed for assigned tasks)
        # elif stuck_agent_ids and time_since_last_gcbba >= min_cooldown:
        #     gcbba_rerun = True  # Trigger GCBBA rerun if there are stuck agents and cooldown has passed

        return OrchestratorEvents(
            completed_task_ids=completed_task_ids,
            stuck_agent_ids=stuck_agent_ids,
            gcbba_rerun=gcbba_rerun
        )

    #################### Energy and Charging Logic (Optional) ####################
    def _get_nearest_charging_station(self, agent_state: AgentState) -> Tuple[int, Tuple[int, int, int]]:
        """
        Returns (bfs_distance, grid_pos) for the nearest charging station to the agent.
        Uses precomputed BFS distance tables (keyed by station grid position).
        Multiple agents may share the same charging station.
        """
        agent_pos = agent_state.get_position()
        best_dist = float('inf')
        best_pos = None

        for station_grid_pos in self.charging_station_grid_positions:
            dist_map = self.grid_map.bfs_distances_from_station.get(station_grid_pos, {})
            dist = dist_map.get(agent_pos, float('inf'))
            if dist < best_dist:
                best_dist = dist
                best_pos = station_grid_pos

        return int(best_dist) if best_dist != float('inf') else 0, best_pos

    def _check_and_start_charging(self) -> List[int]:
        """
        Check if any agents need to start charging and update their state accordingly.
        Multiple agents may share the same physical charging station, but concurrent
        navigating+charging agents are capped at the number of charging stations to
        prevent all agents flooding the narrow approach corridor simultaneously, which
        would make the time-expanded A* planner extremely slow.
        """
        max_concurrent = len(self.charging_station_grid_positions)
        currently_active = sum(
            1 for ast in self.agent_states
            if ast.is_charging or ast.is_navigating_to_charger
        )

        newly_charging_agents = []

        for agent_state in self.agent_states:
            if agent_state.is_charging or agent_state.is_navigating_to_charger or agent_state.is_idle:
                continue

            if currently_active >= max_concurrent:
                break  # Enough agents are already heading to / at chargers this tick

            dist, nearest_charging_station = self._get_nearest_charging_station(agent_state)

            if nearest_charging_station is None:
                continue

            if agent_state.needs_charging(dist):
                agent_state.start_charging(nearest_charging_station)
                newly_charging_agents.append(agent_state.agent_id)
                currently_active += 1

        return newly_charging_agents
    
if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(PROJECT_ROOT, "..", "config", "gridworld_warehouse_small.yaml")

    orchestrator = IntegrationOrchestrator(config_path)

    t0 = time.time()
    orchestrator.run_simulation(timesteps=200)
    tf = time.time()

    print(f"Simulation completed in {tf - t0} seconds.")
    orchestrator.save_trajectories()
