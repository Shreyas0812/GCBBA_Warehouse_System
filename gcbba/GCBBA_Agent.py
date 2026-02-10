"""
GCBBA Agent class for warehouse task allocation
"""

import numpy as np
from math import inf
import copy
import time

class GCBBA_Agent:
    """
    GCBBA Agent for warehouse operations
    """
    def __init__(self, id, G, char_a, tasks, Lt=2, start_time=0, metric="RPT", D=1):
        # int, id of agent
        self.id = id
        # communication matrix G (symmetrical), size Na * Na
        self.G = G
        # int, nb of agents
        self.na = G.shape[0]
        # int, number of neighbors according to G
        self.nb_neigh = np.sum(self.G[id, :])
        # tuple, position in cartesian space
        self.pos = np.array([char_a[0], char_a[1], char_a[2]])
        self.speed = char_a[3]
        self.agent_id = char_a[4]  # Unique agent identifier from warehouse config
        # list of tasks
        self.tasks = tasks
        # int, nb of tasks
        self.nt = len(tasks)
        # capacity
        self.Lt = Lt
        self.D = D
        
        self.metric = metric
        if self.metric == "RPT":
            self.min_val = -1e20
        elif metric == "TDR":
            self.min_val = 0
        
        # list of winning bids for each task (size Nt)
        self.y = [self.min_val for _ in range(self.nt)]
        # list of winners for each task (size Nt)
        self.z = [None for _ in range(self.nt)]
        self.z_before = [None for _ in range(self.nt)]
        # list of bids on each task (size Nt)
        self.c = [self.min_val for _ in range(self.nt)]
        
        # bundle
        self.b = []
        # path /ordered bundle
        self.p = []

        self.S = [0] # Store path scores over time
        
        # timestamps
        self.s = [-inf for _ in range(self.na)]
        self.s[self.id] = 0
        # clock start time
        self.start_time = start_time
        self.converged = False
        
        # convergence observation list
        self.their_net_cvg = [False for _ in range(self.D)]
        self.cvg_counter = 0

        # # Marginal gain list
        # self.Wa = []
        # # List to maintain tasks before insertion
        self.placement = []
        # Has agent won previous bid? (used for reusing previous path)
        self.flag_won = True
        # # size of path at previous iteration
        self.len_p_before = 0

        # task_id to index mapping
        self.task_id_to_idx = {task.id: i for i, task in enumerate(self.tasks)}

    def _get_task_index(self, task_id):
        """Get the index for a given task_id"""
        return self.task_id_to_idx.get(task_id, None)

    def create_bundle(self):
        
        if len(self.p) >= self.Lt: # Check if bundle is not full already
            return
        
        filtered_task_ids = [t.id for t in self.tasks if t.id not in self.p]
        
        if self.flag_won == True:
            self.placement = np.zeros(self.nt)
            for task_id in filtered_task_ids:
                task_idx = self._get_task_index(task_id)
                c, opt_place = self.compute_c(task_id) # c_ij(p_i) = S_i(p_i ⊕_opt j) - S_i(p_i)
                self.c[task_idx] = c
                self.placement[task_idx] = opt_place
        
        # Retain previous bids if agent did not win last time
        
        bids = []
        for j in range(self.nt):
            task_id = self.tasks[j].id

            if task_id not in filtered_task_ids:
                bids.append(self.min_val)
                continue
            
            if self.c[j] > self.y[j]:
                bids.append(self.c[j])
            elif self.c[j] == self.y[j] and self.z[j] is not None and self.z[j] > self.id:
                bids.append(self.c[j])
            else:
                bids.append(self.min_val)
        
        best_idx = np.argmax(bids) # task id with highest bid
        best_task_id = self.tasks[best_idx].id
    
        if best_task_id in self.p or bids[best_idx] <= self.min_val:
            return  # No valid task to add
        
        self.b.append(best_task_id)
        self.p.insert(int(self.placement[best_idx]), best_task_id)
        self.S.append(self.evaluate_path(self.p))  # Update timestamp for this new task addition
        
        self.y[best_idx] = self.c[best_idx]
        self.z[best_idx] = self.id
    
    def compute_c(self, task_id):
        """
        Compute the bid for a given task based on current path
        :param task_id: id of the task to compute bid for
        :return: bid value, optimal placement index

        c^RPT_ij(p_i) = X - min_{1≤l≤|p_i|+1} [L_i(p_i ⊕_l j)^α · W_i(p_i ⊕_l j)^β]
        X is implicit
        alpha = 1, beta = 0 for RPT
        L_i(p): Completion time of entire path p (Equation 13)
        """
        path_bids = []
        P = self.p # List of currrent task ids in path
        
        for pos in range(len(self.p) + 1):
            P1 = copy.deepcopy(P)
            P1.insert(pos, task_id)
            path_score = self.evaluate_path(P1)

            path_bids.append(path_score)
        
        max_bid = np.max(path_bids)
        optimal_pos = np.argwhere(path_bids == max_bid)[-1][0]
        
        return max_bid, optimal_pos

    def evaluate_path(self, path):
        """
        Evaluate the score of a given path based on the selected metric
        :param path: list of tasks in the path
        :return: score of the path -- total path time or makespan
        
        Path now consists of:
        1. Travel to first task's induct position (task.pos[0], task.pos[1]) from agents current position self.pos
        2. For each task: duration is distance from induct (task.pos[0], task.pos[1]) to eject (task.pos[2], task.pos[3])
        3. Current position updates to eject position after each task

        L_i([j^i_1, ..., j^i_n]) = τ_{i,j^i_n}([j^i_1, ..., j^i_n])
        S_i(p_i) = Σ_{j∈p_i} c_ij(p^:j_i)
        """
        cur_pos = self.pos
        score = 0
        time = 0

        if len(path) > 0:
            for j in range(len(path)):
                task_id = path[j]
                task_idx = self._get_task_index(task_id)
                task = self.tasks[task_idx]

                # Later TODO: The task duration will be given by external system which calculates the time as well as the actual collision-free path (we are only concerned with duration tho)
                # Later TODO: May add an additional time to perform the induct/eject operation itself

                # Travel to induct position
                induct_pos = task.induct_pos
                time += np.linalg.norm(cur_pos - induct_pos) / self.speed
                
                # Task duration: distance from induct to eject
                eject_pos = task.eject_pos
                time += np.linalg.norm(induct_pos - eject_pos) / self.speed
                
                score -= time
                time = 0  # Reset time for next task

                # Update current position to eject position
                cur_pos = eject_pos
                
        return score
    
    def resolve_conflicts(self, all_agents, consensus_iter=0, consensus_index_last=False):
        """
        Resolve conflicts with neighboring agents based on communication graph G -- consensus phase
        :param all_agents: list of all agents in the system
        :param consensus_iter: current consensus iteration number
        :param consensus_index_last: boolean indicating if this is the last consensus index
        """
        niegh_idxs = np.argwhere(self.G[self.id, :] == 1).flatten()
        niegh_idxs = niegh_idxs[niegh_idxs != self.id]  # Exclude self

        neigh_cvg = [True for _ in range(self.D)]

        for k in niegh_idxs:
            neigh = all_agents[k]

            for j in range(self.nt):
                task_id = self.tasks[j].id

                # agent k (neighbour) thinks it won task j
                if neigh.z[j] == neigh.id:
                    # agent i (self) thinks it won task j
                    if self.z[j] == self.id:
                        # conflict: both agents think they won task j
                        if neigh.y[j] > self.y[j] or (neigh.y[j] == self.y[j] and neigh.id < self.id):
                            self.update(neigh, task_id)
                    # agent i (self) thinks agent k (neighbour) won task j
                    elif self.z[j] == k:
                        self.update(neigh, task_id)
                    # agent i (self) thinks no one won task j
                    elif self.z[j] == None:
                        self.update(neigh, task_id)
                    # agent i (self) thinks agent m (neighbour m != k) won task j
                    else:
                        m = int(self.z[j])
                        # update only if neighbour has more recent info
                        if neigh.s[m] > self.s[m] or neigh.y[j] > self.y[j] or (neigh.y[j] == self.y[j] and neigh.id < self.id):
                            self.update(neigh, task_id)
                
                # agent k (neighbour) thinks agent i (self) won task j
                elif neigh.z[j] == self.id:
                    # agent i (self) thinks it won task j -- leave as is
                    if self.z[j] == self.id:
                        self.leave()
                    # agent i (self) thinks agent k (neighbour) won task j
                    elif self.z[j] == k:
                        self.reset(task_id) # reset task j to clear conflict
                    # agent i (self) thinks no one won task j
                    elif self.z[j] == None:
                        self.leave()
                    # agent i (self) thinks agent m (neighbour m != k) won task j
                    else:
                        m = int(self.z[j])
                        # update only if neighbour has more recent info
                        if neigh.s[m] > self.s[m]:
                            self.reset(task_id)

                # agent k (neighbour) thinks no one won task j
                elif neigh.z[j] == None:
                    # agent i (self) thinks it won task j
                    if self.z[j] == self.id:
                        self.leave()
                    # agent i (self) thinks agent k (neighbour) won task j
                    elif self.z[j] == k:
                        self.update(neigh, task_id)
                    # agent i (self) thinks no one won task j
                    elif self.z[j] == None:
                        self.leave()
                    # agent i (self) thinks agent m (neighbour m != k) won task j
                    else:
                        m = int(self.z[j])
                        # update only if neighbour has more recent info
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, task_id)
                
                # agent k (neighbour) thinks agent m (neighbour m != k) won task j
                else:
                    m = int(neigh.z[j])
                    # agent i (self) thinks it won task j
                    if self.z[j] == self.id:
                        # update only if neighbour has more recent info
                        if (neigh.s[m] > self.s[m] and neigh.y[j] > self.y[j]) or (neigh.s[m] > self.s[m] and neigh.y[j] == self.y[j] and neigh.id < self.id):
                            self.update(neigh, task_id)
                    # agent i (self) thinks agent k (neighbour) won task j
                    elif self.z[j] == k:
                        # neighbour has more recent info
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, task_id)
                        # reset stale belief
                        else:
                            self.reset(task_id)
                    # agent i (self) also thinks agent m won task j
                    elif self.z[j] == m:
                        # update only if neighbour has more recent info
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, task_id)
                    # agent i (self) thinks no one won task j
                    elif self.z[j] == None:
                        # update only if neighbour has more recent info
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, task_id)
                    # agent i (self) thinks agent n (neighbour n != k, n != m) won task j
                    else:
                        n = int(self.z[j])
                        # If neighbor has fresher info about BOTH m and n → accept neighbor's view (update)
                        if neigh.s[m] > self.s[m] and neigh.s[n] > self.s[n]:
                            self.update(neigh, task_id)
                        # If neighbor has fresher info about m AND higher bid → update
                        elif (neigh.s[m] > self.s[m] and neigh.y[j] > self.y[j]) or (neigh.s[m] > self.s[m] and neigh.y[j] == self.y[j] and neigh.id < self.id):
                            self.update(neigh, task_id)
                        # If neighbor has fresher info about n but you have fresher info about m → conflict, reset
                        elif neigh.s[n] > self.s[n] and self.s[m] > neigh.s[m]:
                            self.reset(task_id)
            self.compute_s(neigh, consensus_iter)

            # Neighbor convergence observation update
            for i in range(1, self.D):
                neigh_cvg[i] = neigh_cvg[i] and neigh.their_net_cvg[i-1]
            
        # Update convergence observation
        self.their_net_cvg[0] = (self.z == self.z_before)
        for i in range(1, self.D):
            self.their_net_cvg[i] = neigh_cvg[i] and self.their_net_cvg[i-1]
        
        self.converged = self.their_net_cvg[-1]

        if self.converged:
            self.cvg_counter += 1
        
        if consensus_index_last:
            self.z_before = copy.deepcopy(self.z)
            self.flag_won = (len(self.p) != self.len_p_before)
            self.len_p_before = len(self.p)


    def update(self, neighbor, task_id):

        task_idx = self._get_task_index(task_id)

        if task_idx is None:
            print(f"Warning: task_id {task_id} not found in task list")
            return

        # accept neighbor's bid and winner for task_id
        self.y[task_idx] = neighbor.y[task_idx]
        self.z[task_idx] = neighbor.z[task_idx]

        bundle = self.b
        # if task is in own bundle, remove it and all subsequent tasks
        if task_id in bundle:                           
            self.flag_won = False

            # position of task in bundle, remove from there onwards
            bundle_index = bundle.index(task_id)
            tasks_to_remove = bundle[bundle_index:]

            # clear winning bids and winners for removed tasks
            for task_id_to_remove in tasks_to_remove:
                idx = self._get_task_index(task_id_to_remove)
                self.y[idx] = self.min_val
                self.z[idx] = None

            # accept neighbor's bid and winner for task_id again to ensure consistency in case it was removed
            self.y[task_idx] = neighbor.y[task_idx]
            self.z[task_idx] = neighbor.z[task_idx]

            # remove tasks from bundle and path
            self.b = self.b[:bundle_index]

            for task in tasks_to_remove:
                if task in self.p:
                    self.p.remove(task)

            self.S = self.S[:bundle_index + 1]

            self.their_net_cvg[0] = False  # Reset convergence observation since bundle changed

    def reset(self, task_id):
        task_idx = self._get_task_index(task_id)

        if task_idx is None:
            print(f"Warning: task_id {task_id} not found in task list")
            return
        
        # Clear own bid and winner for task_id
        self.y[task_idx] = self.min_val
        self.z[task_idx] = None

        bundle = self.b
        # if task is in own bundle, remove it and all subsequent tasks
        if task_id in bundle:
            bundle_index = bundle.index(task_id)
            tasks_to_remove = bundle[bundle_index:]

            for task_id_to_remove in tasks_to_remove:
                idx = self._get_task_index(task_id_to_remove)
                self.y[idx] = self.min_val
                self.z[idx] = None
                    
            self.b = self.b[:bundle_index]

            for task in tasks_to_remove:
                if task in self.p:
                    self.p.remove(task)

            self.S = self.S[:bundle_index + 1]

            self.their_net_cvg[0] = False  # Reset convergence observation since bundle changed
    
    def leave(self):
        # print (f"Agent {self.id} doing nothing this round")
        pass
    
    def compute_s(self, neighbor, consensus_iter):
        """
        Update timestamps based on neighbor's information
        :param neighbor: neighboring agent
        :param consensus_iter: current consensus iteration number
        """
        self.s[self.id] = consensus_iter
        self.s[neighbor.id] = consensus_iter

        not_neighbor_ids = np.argwhere(self.G[self.id, :] == 0).flatten()
        greater_index = np.argwhere(np.array(neighbor.s) > np.array(self.s)).flatten()
        intersect = list(set(greater_index).intersection(set(not_neighbor_ids)))

        self.s = [neighbor.s[i] if i in intersect else self.s[i] for i in range(self.na)]
