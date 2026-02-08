import yaml
import os
from math import ceil
import time
import numpy as np
import networkx as nx

from tools_warehouse import create_graph_with_range, agent_init, random_task_init, task_init
from GCBBA_Orchestrator import GCBBA_Orchestrator

def read_warehouse_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":

    seed = 5 # Random seed for reproducibility
    np.random.seed(seed)

    # Get the path to the config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config', 'gridworld_warehouse_small.yaml')
    
    # Read the configuration
    config = read_warehouse_config(config_path)
    
    # Access the parameters
    params = config['create_gridworld_node']['ros__parameters']
    
    # GCBBA specific parameters
    na = len(params['agent_positions']) // 3  # Number of agents
    tasks_per_induct_station = 10  # Number of tasks per induct station
    xlim = [0, int(params['grid_width']) * params['grid_resolution']]
    ylim = [0, int(params['grid_height']) * params['grid_resolution']]
    sp_lim = [1, 5] # Speed limits in units/sec
    dur_lim = [1, 10] # Task duration limits in seconds
    comm_range = 30  # Communication range in units
    
    # Extract agent positions from config (reshape flattened array)
    agent_pos_flat = params['agent_positions']
    # LATER TODO: add agent IDs from config
    # Add agent_id to agent positions and then use the commented line instead
    agent_positions = [(agent_pos_flat[i], agent_pos_flat[i+1], agent_pos_flat[i+2], i//3 + 1) 
                       for i in range(0, len(agent_pos_flat), 3)]
    # agent_positions = [(agent_pos_flat[i], agent_pos_flat[i+1], agent_pos_flat[i+2]) 
    #                    for i in range(0, len(agent_pos_flat), 4)]
    

    # Extract induct station positions from config
    induct_pos_flat = params['induct_stations']
    induct_positions = [(induct_pos_flat[i], induct_pos_flat[i+1], induct_pos_flat[i+2], induct_pos_flat[i+3]) 
                        for i in range(0, len(induct_pos_flat), 4)]
    
    # Extract induct station positions from config (these become tasks)
    induct_pos_flat = params['induct_stations']
    induct_positions = [(induct_pos_flat[i], induct_pos_flat[i+1], induct_pos_flat[i+2], induct_pos_flat[i+3]) 
                        for i in range(0, len(induct_pos_flat), 4)] 

    # Extract eject station positions from config (these become tasks)
    eject_pos_flat = params['eject_stations']
    eject_positions = [(eject_pos_flat[i], eject_pos_flat[i+1], eject_pos_flat[i+2], eject_pos_flat[i+3]) 
                       for i in range(0, len(eject_pos_flat), 4)]
    
    nt = len(induct_positions) * tasks_per_induct_station  # Number of tasks 
    Lt = ceil(nt / na)  # Tasks per agent

    # Later TODO: Create communication graph based on distance (agent-to-agent and agent-to-induct)
    # raw_graph, G = create_graph_with_range(agent_positions, induct_positions, comm_range)
    
    # Create communication graph based on distance (agent-to-agent only)
    raw_graph, G = create_graph_with_range(agent_positions, comm_range)
    
    D = nx.diameter(raw_graph)
    
    # Initialize agents and tasks from warehouse config
    agents = agent_init(agent_positions, sp_lim=sp_lim)
    tasks = task_init(induct_positions, eject_positions, task_per_induct_station=tasks_per_induct_station)

    # Initialize orchestrator (placeholder for now)
    orch_cbba = GCBBA_Orchestrator(G, D, tasks, agents, Lt)

    t0 = time.time()
    assig, tot_score, makespan = orch_cbba.launch_agents()
    tf0 = np.round(1000 * (time.time() - t0))


    print("GCBBA - total score. = {}; max score = {}; time = {} ms; assignment = {}".format(tot_score, makespan, tf0, assig))

    


    
