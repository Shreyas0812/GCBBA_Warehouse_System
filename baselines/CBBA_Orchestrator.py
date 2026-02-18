"""
CBBA Orchestrator for warehouse task allocation.

Standard CBBA (Choi et al. 2009) baseline:
- FULLBUNDLE bundle building (agents fill entire bundle before consensus)
- 1 consensus round per iteration
- Nmin * D total iterations
- No convergence detection

Interface matches GCBBA_Orchestrator for drop-in replacement.
"""

import time
from gcbba.GCBBA_Task import GCBBA_Task
from baselines.CBBA_Agent import CBBA_Agent

class CBBA_Orchestrator:
    """
    Standard CBBA Orchestrator for warehouse task allocation.
    """
    def __init__(self, G, D, char_t, char_a, Lt=1, metric="RPT", task_ids=None, grid_map=None):
        self.G = G
        self.D = D
        self.char_t = char_t
        self.char_a = char_a
        self.Lt = Lt
        self.metric = metric
        self.task_ids = task_ids if task_ids is not None else list(range(len(char_t)))
        self.grid_map = grid_map

        self.na = G.shape[0]
        self.nt = len(char_t)

        self.start_time = time.perf_counter()

        self.agents = []
        self.tasks = []

        self.initialize_all()

        self.bid_history = []
        self.assig_history = []
        self.max_times = []
    
    def initialize_all(self):
        self.initialize_tasks()
        self.initialize_agents()

    def initialize_tasks(self):
        self.tasks = []
        for j in range(self.nt):
            self.tasks.append(GCBBA_Task(id=self.task_ids[j], char_t=self.char_t[j], grid_map=self.grid_map))

    def initialize_agents(self):
        self.agents = []
        for i in range(self.na):
            self.agents.append(CBBA_Agent(id=i, G=self.G, char_a=self.char_a[i], tasks=self.tasks, Lt=self.Lt, start_time=self.start_time, metric=self.metric, D=self.D, grid_map=self.grid_map))

    def launch_agents(self, method=None, detector=None):
        """
        Launch CBBA

        - nb_iter = Nmin * D total iterations
        - nb_consensus = 1 consensus round per iteration
        - FULLBUNDLE bundle building (agents fill entire bundle before consensus)
        - no convergence detection
        """

        D = self.D
        Nmin = int(min(self.na * self.Lt, self.nt))

        nb_iter = Nmin * D
        nb_cons = 1

        for iteration in range(nb_iter):
            # Phase 1: Bundle Building
            for agent in self.agents:
                agent.create_bundle()
            
            # Phase 2: Consensus (single round)
            for consensus_num in range(nb_cons):
                all_agents = [agent.snapshot() for agent in self.agents]
                consensus_iter = nb_cons * iteration + consensus_num

                for agent in self.agents:
                    agent.resolve_conflicts(all_agents, consensus_iter=consensus_iter, consensus_index_last=True)
            
            assignment, bid, max_time = self.gather_info()

            self.assig_history.append(assignment)
            self.bid_history.append(bid)
            self.max_times.append(max_time)
    
        if len(self.assig_history) > 0:
            return self.assig_history[-1], self.bid_history[-1], self.max_times[-1]
        else:
            return [], 0, 0