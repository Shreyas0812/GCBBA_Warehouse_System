import networkx as nx
import numpy as np
import time
from tqdm import tqdm
import copy

from GCBBA_Task import GCBBA_Task
from GCBBA_Agent import GCBBA_Agent

class GCBBA_Orchestrator:
    """
    GCBBA Orchestrator for warehouse task allocation
    """
    def __init__(self, G, D, char_t, char_a, Lt=1, metric="RPT"):
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
        # list of all agents
        self.agents = []
        # list of all tasks
        self.tasks = []
        
        # clock launch
        self.start_time = time.perf_counter()
        
        self.metric = metric
        self.D = D
        
        # initialize tasks and agents
        self.initialize_all()
        print("Orchestrator initialized with {} agents and {} tasks.".format(self.na, self.nt))
        self.bid_history = []
        self.assig_history = []
        self.max_times = []
        self.all_times = [0 for _ in range(self.na)]
        
        self.cvg_iter = self.nt
    
    def initialize_all(self):
        self.initialize_tasks()
        self.initialize_agents()
    
    def initialize_tasks(self):
        self.tasks = []
        for j in range(self.nt):
            char_t = self.char_t[j]
            self.tasks.append(GCBBA_Task(id=j, char_t=char_t))
    
    def initialize_agents(self):
        self.agents = []
        for i in range(self.na):
            char_a = self.char_a[i]
            self.agents.append(
                GCBBA_Agent(id=i, G=self.G, char_a=char_a, tasks=self.tasks, Lt=self.Lt, 
                           start_time=self.start_time, metric=self.metric, D=self.D))
    
    def launch_agents(self, method="global", detector="decentralized"):
        """
        Launch GCBBA allocation algorithm
        :param method: "baseline" or "global"
        :param detector: "none", "centralized", or "decentralized"
        :return: allocation, minsum, makespan
        """
        D = self.D
        Nmin = int(min(self.nt, self.Lt * self.na)) # minimum number of assigned tasks

        nb_iter = Nmin # number of main iterations
        nb_cons = 2*D  # number of consensus rounds per iteration

        for iter in tqdm(range(nb_iter)):
            for i in range(self.na):
                # Bundle creation phase
                if self.agents[i].converged == False:
                    self.agents[i].create_bundle()
            
            # Consensus phase
            for consensus_num in range(nb_cons):
                all_agents = copy.deepcopy(self.agents)
                consensus_iter = nb_cons * iter + consensus_num
                if consensus_num == nb_cons - 1:
                    consensus_index_last = True
                else:
                    consensus_index_last = False
                
                for i in range(self.na):
                    if self.agents[i].converged == False:
                        self.agents[i].resolve_conflicts(all_agents, consensus_iter=consensus_iter, consensus_index_last=consensus_index_last)
            
            assignment, bid, max_time = self.gather_info()
            self.assig_history.append(assignment)
            self.bid_history.append(bid)
            self.max_times.append(max_time)

            all_converged = np.all([agent.converged for agent in self.agents])
            if all_converged and self.cvg_iter == self.nt:
                self.cvg_iter = iter + 1
                print("All agents converged at iteration {}".format(self.cvg_iter))
                break

            # if not np.all([agent.converged for agent in self.agents]):
            #     self.cvg_iter = iter + 1

        if len(self.assig_history) > 0:
            return self.assig_history[-1], self.bid_history[-1], self.max_times[-1]
        else:
            return [], 0, 0
    
    def gather_info(self):
        """
        Gather assignment and bid information from all agents
        :return: assignment list, minsum and makespan
        """
        bid_sum = 0
        assignment = []
        max_time = 0

        for i in range(self.na):
            agent = self.agents[i]
            a_time = agent.evaluate_path(agent.p)
            a_time = -a_time  # since bids are negative times

            if a_time > max_time:
                max_time = a_time
            
            bid_sum += a_time
            assignment.append(copy.deepcopy(agent.p))
        
        return assignment, np.round(bid_sum, 6), max_time
    
