import heapq
import os
import sys
from collections import defaultdict

from collision_avoidance.grid_map import GridMap

class TimeBasedCollisionAvoidance:
    """
    TimeBasedCollisionAvoidance for collision free MAPF
    """

    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.reservations = {} # {(x, y, z, t): agent_id}

    def clear_agent_reservations(self, agent_id):
        keys_to_remove = [k for k, v in self.reservations.items() if v == agent_id]
        for k in keys_to_remove:
            del self.reservations[k]
            
    def clear_all_reservations(self):
        """
        Clears all reservations.
        """
        self.reservations = {}
    
    def heuristic(self, pos1, pos2):
        """
        Computes the Manhattan distance heuristic between two positions.
        
        :param pos1: (x1, y1, z1)
        :param pos2: (x2, y2, z2)
        :return: Manhattan distance
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
    
    def is_reserved(self, x, y, z, t, agent_id):
        """
        Checks if a cell is reserved at a given time for a different agent.
        
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :param t: time step
        :param agent_id: ID of the agent checking the reservation
        :return: True if reserved by another agent, False otherwise
        """
        key = (x, y, z, t)
        if key in self.reservations:
            return self.reservations[key] != agent_id
        return False
    
    def has_edge_conflict(self, curr_pos, next_pos, t, agent_id):
        """
        Checks for edge conflicts (swapping positions) between two agents.
        
        :param curr_pos: Current position of the agent (x1, y1, z1)
        :param next_pos: Next position of the agent (x2, y2, z2)
        :param t: current time step
        :param agent_id: ID of the agent checking for conflicts
        :return: True if there is an edge conflict, False otherwise
        """
        if self.is_reserved(*next_pos, t-1, agent_id) and self.is_reserved(*curr_pos, t, agent_id):
            # Another agent is moving from next_pos to curr_pos at the same time
            # Basically a swap conflict
            return True
        return False
    
    def reserve_path(self, path, agent_id, start_time=0):
        """
        Reserves the path for the agent.
        
        :param path: List of positions [(x1, y1, z1), (x2, y2, z2), ...]
        :param agent_id: ID of the agent
        """
        for t, pos in enumerate(path):
            key = (*pos, t + start_time)
            self.reservations[key] = agent_id

        # Reserve goal position for all future timesteps 
        if path:
            goal_pos = path[-1]
            for future_t in range(len(path) + start_time, len(path) + start_time + 1000): # Arbitrary large number to reserve goal
                key = (*goal_pos, future_t)
                self.reservations[key] = agent_id
    
    def plan_path_with_reservations(self, start, goal, agent_id, max_time=1000, start_time=0):
        """
        Path plan with A* search considering time-based reservations.

        :param start: Starting position (x, y, z)
        :param goal: Goal position (x, y, z)
        :param agent_id: ID of the agent
        :param max_time: Maximum time steps to search
        :param start_time: Starting time step for planning
        :return: List of positions [(x1, y1, z1), (x2, y2, z2), ...] or None if no path found
        """

        if start == goal:
            return [start]
        
        if not self.grid_map.is_valid_cell(*start) or not self.grid_map.is_valid_cell(*goal):
            return None
        
        # Priority queue for A* search: (f_score, counter, (position, time_step), path)
        counter = 0
        open_set = [(0, counter, (start, start_time), [start])]
        closed_set = set()

        while open_set:
            f_score, _, (current_pos, current_time), path = heapq.heappop(open_set)

            if (current_pos, current_time) in closed_set:
                continue

            if current_pos == goal:
                return path
            
            if current_time >= max_time:
                continue

            closed_set.add((current_pos, current_time))

            # Wait in place option
            next_time = current_time + 1
            if not self.is_reserved(*current_pos, next_time, agent_id):
                path_new = path + [current_pos]
                g_score = len(path_new) - 1
                h_score = self.heuristic(current_pos, goal)
                f_score = g_score + h_score
                counter += 1

                heapq.heappush(open_set, (f_score, counter, (current_pos, next_time), path_new))

            # Move to neighbors
            for neighbor in self.grid_map.get_neighbors(*current_pos):
                

                if ((neighbor, next_time) in closed_set) or (self.is_reserved(*neighbor, next_time, agent_id)) or (self.has_edge_conflict(current_pos, neighbor, next_time, agent_id)):
                    continue

                path_new = path + [neighbor]
                g_score = len(path_new) - 1
                h_score = self.heuristic(neighbor, goal)
                f_score = g_score + h_score
                counter += 1

                heapq.heappush(open_set, (f_score, counter, (neighbor, next_time), path_new))

        return None
    
    def plan_all_agents(self, agents, goals, priorities=None):
        """
        Plans paths for all agents considering time-based reservations.

        :param agents: List of starting positions [(x1, y1, z1), (x2, y2, z2), ...]
        :param goals: List of goal positions [(x1, y1, z1), (x2, y2, z2), ...]
        :param priorities: Optional list of agent IDs in order of priority (lower index = higher priority)
        :return: Dictionary {agent_id: path} where path is a list of positions or None if no path found
        """
        if priorities is None:
            priorities = list(range(len(agents)))

        agent_paths = {}
        for idx in priorities:
            agent = agents[idx]
            agent_id = agent.id

            start = self.grid_map.continuous_to_grid(agent.pos[0], agent.pos[1], agent.pos[2])

            if idx not in goals and agent_id not in goals:
                # Agent has no goal, just reserve its current position
                agent_paths[agent_id] = [start]
                self.reserve_path(agent_paths[agent_id], agent_id)
                continue

            goal = goals[idx] if idx in goals else goals[agent_id]

            # Plan path for the agent
            path = self.plan_path_with_reservations(start, goal, agent_id)

            if path:
                agent_paths[agent_id] = path
                self.reserve_path(path, agent_id)
            else:
                # No Path, agent will wait in place indefinitely
                agent_paths[agent_id] = [start]
                self.reserve_path(agent_paths[agent_id], agent_id)

        return agent_paths
    

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    class SimpleAgent:
        def __init__(self, agent_id, pos):
            self.id = agent_id
            self.pos = pos

    def check_collisions(path1, path2):
        collisions = 0
        max_len = max(len(path1), len(path2))

        p1 = path1 + [path1[-1]] * (max_len - len(path1))
        p2 = path2 + [path2[-1]] * (max_len - len(path2))

        for t in range(max_len):
            if p1[t] == p2[t]:
                collisions += 1
                print(f"  Collision at time {t}: both at {p1[t]}")

        return collisions

    print("\n--- Example: Head-On Collision ---")
    config_path = os.path.join(PROJECT_ROOT, "config", "gridworld_warehouse_small.yaml")
    grid_map = GridMap(config_path)
    ca = TimeBasedCollisionAvoidance(grid_map)

    agents = [
        SimpleAgent(0, [5, 4, 0]),
        SimpleAgent(1, [3, 4, 0])
    ]
    goals = {
        0: grid_map.continuous_to_grid(3, 4, 0),
        1: grid_map.continuous_to_grid(5, 4, 0)
    }

    paths = ca.plan_all_agents(agents, goals)

    print(f"Agent 0 path: {paths[0][:5]}...")
    print(f"Agent 1 path: {paths[1][:5]}...")

    collisions = check_collisions(paths[0], paths[1])
    if collisions == 0:
        print("✓ No collisions detected")
    else:
        print(f"✗ {collisions} collisions detected!")
   