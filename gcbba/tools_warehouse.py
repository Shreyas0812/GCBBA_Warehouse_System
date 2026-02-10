"""
File of auxiliary functions
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import cycle


# Create a communication graph including only agents
def create_graph_with_range(agent_positions, comm_range):
    """
    Creates a communication graph based on distance between agents.
    Connections are created if agents are within communication range.
    
    :param agent_positions: List of agent positions [(x1, y1, z1, id1), (x2, y2, z2, id2), ...]
    :param comm_range: Maximum communication range (distance threshold)
    :return: raw networkx graph, corresponding communication matrix G
    """
    # Extract agent IDs and create node labels
    agent_ids = [f"agent_{int(pos[3])}" for pos in agent_positions]
    n_agents = len(agent_ids)
    
    # Create empty graph with labeled nodes
    raw_graph = nx.Graph()
    raw_graph.add_nodes_from(agent_ids)
    
    # Helper function to calculate 3D Euclidean distance
    def distance(pos1, pos2):
        return np.sqrt(
            (pos1[0] - pos2[0])**2
            + (pos1[1] - pos2[1])**2
            + (pos1[2] - pos2[2])**2
        )
    
    # Add edges between agents if within communication range
    for i, agent_id_i in enumerate(agent_ids):
        for j, agent_id_j in enumerate(agent_ids[i + 1:], start=i + 1):
            if distance(agent_positions[i], agent_positions[j]) <= comm_range:
                raw_graph.add_edge(agent_id_i, agent_id_j)
    
    # Convert to adjacency matrix (nodes will be ordered as in agent_ids)
    G = nx.adjacency_matrix(raw_graph, nodelist=agent_ids).todense()
    G = np.asarray(G.astype('float'))
    np.fill_diagonal(G, 1.0)  # Ensure self-connections
    
    return raw_graph, G


def agent_init(agent_positions, sp_lim=[1, 5]):
    """
    Create agents characteristics from warehouse configuration
    :param agent_positions: List of agent positions [(x, y, z, id), ...]
    :param sp_lim: agent speed limits
    :return: agents list
    """
    # Agents: [x_pos, y_pos, z_pos, speed, id]
    agents = []
    for pos in agent_positions:
        speed = np.random.uniform(sp_lim[0], sp_lim[1], 1)
        agents.append(np.concatenate(([pos[0], pos[1], pos[2]], speed, pos[-1:])))
    
    return agents


def task_init(induct_positions, eject_positions, task_per_induct_station):
    """
    Create tasks characteristics from warehouse configuration
    :param induct_positions: List of induct station positions [(x, y, z, id), ...] - these become tasks
    :param eject_positions: List of eject station positions [(x, y, z, id), ...] - these become tasks
    :param task_per_induct_station: Number of tasks per induct station
    :return: tasks list
    """
    # Tasks: [induct_x, induct_y, induct_z, eject_x, eject_y, eject_z]
    tasks = []
    for current_induct_position in induct_positions:
        for _ in range(task_per_induct_station):
            random_eject_position = eject_positions[np.random.randint(0, len(eject_positions))]
            tasks.append(
                np.array(
                    [
                        current_induct_position[0],
                        current_induct_position[1],
                        current_induct_position[2],
                        random_eject_position[0],
                        random_eject_position[1],
                        random_eject_position[2],
                    ]
                )
            )
    return tasks


# Unused code

def random_agent_init(na = 10, pos_lim = [-5, 5], sp_lim = [1, 5]):
    """
    Create agents characteristics
    :param na: number of agents
    :param pos_lim: (x,y) limits
    :param sp_lim: agent speed limits
    :return: list of agents with characteristics [x_pos, y_pos, speed]
    """
    # Agents: [x_pos, y_pos, speed]
    agents = [np.concatenate(
                (np.random.uniform(pos_lim[0], pos_lim[1], 2), 
                 np.random.uniform(sp_lim[0], sp_lim[1], 1))
                 ) for i in range(na)]          
    
    return agents

def random_task_init(nt =100, xlim = [-5, 5], ylim = [-5, 5], dur_lim = [1,5], lamb_lim = [0.95, 0.95], clim = [1, 1]):
    """
    Create tasks characteristics
    :param nt: number of tasks
    :param xlim: x-axis limits
    :param ylim: y-axis limits
    :param dur_lim: task duration limits
    :param lamb_lim: lambda (TDR) limits
    :param clim: weights (TDR) limits (useless in this implementation)
    :return:
    """
    # Tasks: [x_pos, y_pos, duration, lambda, weight]
    tasks = [np.concatenate(
                (np.random.uniform(xlim[0], xlim[1], 1), 
                 np.random.uniform(ylim[0], ylim[1], 1), 
                 np.random.uniform(dur_lim[0], dur_lim[1], 1), 
                 np.random.uniform(lamb_lim[0], lamb_lim[1], 1),
                np.random.uniform(clim[0], clim[1], 1))
                ) for _ in range(nt)]
    
    return tasks

def create_graph_with_range2(agent_positions, induct_positions, comm_range):
    """
    Creates a communication graph based on distance between agents and induct stations.
    Connections are created if entities are within communication range.
    
    :param agent_positions: List of agent positions [(x1, y1, z1, id1), (x2, y2, z2, id2), ...]
    :param induct_positions: List of induct station positions [(x1, y1, z1, id1), ...]
    :param comm_range: Maximum communication range (distance threshold)
    :return: raw networkx graph, corresponding communication matrix G
    """
    # Extract agent IDs and create node labels
    agent_ids = [f"agent_{int(pos[3])}" for pos in agent_positions]
    induct_ids = [f"induct_{int(pos[3])}" for pos in induct_positions]
    all_node_ids = agent_ids + induct_ids
    
    # Create empty graph with labeled nodes
    raw_graph = nx.Graph()
    raw_graph.add_nodes_from(all_node_ids)
    
    # Helper function to calculate 3D Euclidean distance
    def distance(pos1, pos2):
        return np.sqrt(
            (pos1[0] - pos2[0])**2
            + (pos1[1] - pos2[1])**2
            + (pos1[2] - pos2[2])**2
        )
    
    # Add edges between agents if within communication range
    for i, agent_id_i in enumerate(agent_ids):
        for j, agent_id_j in enumerate(agent_ids[i + 1:], start=i + 1):
            if distance(agent_positions[i], agent_positions[j]) <= comm_range:
                raw_graph.add_edge(agent_id_i, agent_id_j)
    
    # Add edges between agents and induct stations if within communication range
    for i, agent_id in enumerate(agent_ids):
        for j, induct_id in enumerate(induct_ids):
            if distance(agent_positions[i], induct_positions[j]) <= comm_range:
                raw_graph.add_edge(agent_id, induct_id)
    
    # Add edges between induct stations if within communication range
    for i, induct_id_i in enumerate(induct_ids):
        for j, induct_id_j in enumerate(induct_ids[i + 1:], start=i + 1):
            if distance(induct_positions[i], induct_positions[j]) <= comm_range:
                raw_graph.add_edge(induct_id_i, induct_id_j)
    
    # Convert to adjacency matrix (nodes will be ordered as in all_node_ids)
    G = nx.adjacency_matrix(raw_graph, nodelist=all_node_ids).todense()
    
    return raw_graph, np.asarray(G.astype('float'))