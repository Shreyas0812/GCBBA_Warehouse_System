"""
CBBA Agent class for Warehouse Task Allocation.

Standard CBBA (Choi et al. 2009) baseline:
- FULLBUNDLE bundle building: fills entire bundle in a while loop before consensus
- Same consensus/conflict resolution as GCBBA
- No convergence detection
"""

import numpy as np

from gcbba.GCBBA_Agent import GCBBA_Agent

class CBBA_Agent(GCBBA_Agent):
    """
    Same functoins as GCBBA_Agent but overrides create_bundle() to use FULLBUNDLE strategy.
        - GCBBA (ADD): adds ONE task per create_bundle() call, then 2D consensus rounds
        - CBBA (FULLBUNDLE): adds ALL tasks up to capacity in one create_bundle() call, 
        then 1 consensus round

    Similarly in resolve_conflicts(), CBBA only does 1 round of conflict resolution after bundle building, no convergence detection.
    
    Other functionality remaints the same
    """

    def __init__(self, id, G, char_a, tasks, Lt=2, start_time=0, metric="RPT", D=1, grid_map=None):
        super().__init__(id, G, char_a, tasks, Lt, start_time, metric, D, grid_map)
        # CBBA does not use convergence detection
        self.converged = False

    def create_bundle(self):
        """
        FULLBUNDLE strategy: greedily add tasks to bundle until capacity is reached
        or no more tasks can improve the path.
        
        This fills the entire bundle in one call (standard CBBA Phase 1).
        """
        while len(self.p) < self.Lt:
            # Recompute all bids from scratch each time a task is added
            optimal_placement = np.zeros(self.nt)
            filtered_task_ids = [t.id for t in self.tasks if t.id not in self.p]

            if not filtered_task_ids:
                return

            for task_id in filtered_task_ids:
                task_idx = self._get_task_index(task_id)
                c, opt_place = self.compute_c(task_id)
                self.c[task_idx] = c
                optimal_placement[task_idx] = opt_place

            # Select best task that beats current winning bid
            bids = []
            for j in range(self.nt):
                task_id = self.tasks[j].id
                if task_id not in filtered_task_ids:
                    bids.append(self.min_val)
                    continue

                if self.c[j] > self.y[j]:
                    bids.append(self.c[j])
                elif self.c[j] == self.y[j] and self.z[j] is not None and self.id < self.z[j]:
                    bids.append(self.c[j])
                else:
                    bids.append(self.min_val)

            best_bid_idx = np.argmax(bids)
            best_task_id = self.tasks[best_bid_idx].id

            if best_task_id in self.p or bids[best_bid_idx] <= self.min_val:
                break  # No valid task to add, exit loop
            
            # Add best task to bundle and path
            self.b.append(best_task_id)
            self.p.insert(optimal_placement[best_bid_idx], best_task_id)
            self.S.append(self.evaluate_path(self.p))

            self.y[best_bid_idx] = self.c[best_bid_idx]
            self.z[best_bid_idx] = self.id
    
    def resolve_conflicts(self, all_agents, consensus_iter=0, consensus_index_last=False):
        """
        CBBA does not do convergence detection, so just do 1 round of conflict resolution after bundle building.

        Logic remains the same, but there is no convergence detection 
        """
        neigh_indxs = np.argwhere(self.G[self.id, :] == 1).flatten()
        neigh_indxs = neigh_indxs[neigh_indxs != self.id]  # Exclude self

        for k in neigh_indxs:
            neigh = all_agents[k]
            for j in range(self.nt):

                task_id = self.tasks[j].id

                # agent k thinks it won task j
                if neigh.z[j] == neigh.id:
                    # agent i (self) thinks it won task j 
                    if self.z[j] == self.id:
                        if neigh.y[j] > self.y[j] or (neigh.y[j] == self.y[j] and neigh.id < self.id):
                            self.update(neigh, task_id)
                    # agent i (self) thinks k won task j
                    elif self.z[j] == k:
                        self.update(neigh, task_id)
                    # agent i (self) thinks nobody won task j
                    elif self.z[j] is None:
                        self.update(neigh, task_id)
                    # agent j thinks some other agent m won task j
                    else:
                        m = int(self.z[j])
                        if neigh.s[m] > self.s[m] or neigh.y[j] > self.y[j] or (neigh.y[j] == self.y[j] and neigh.id < self.id):
                            self.update(neigh, task_id)

                # agent k thinks agent i (self) won task j
                elif neigh.z[j] == self.id:
                    # agent i (self) also thinks agent i (self) won task j
                    if self.z[j] == self.id:
                        self.leave()
                    # agent i (self) thinks k won task j
                    elif self.z[j] == k:
                        self.reset(task_id)
                    # agent i (self) thinks nobody won task j
                    elif self.z[j] is None:
                        self.leave()
                    # agent i (self) thinks some other agent m won task j
                    else:
                        m = int(self.z[j])
                        # update if neighbor has more recent info
                        if neigh.s[m] > self.s[m]:
                            self.reset(task_id)
                
                # agent k thinks nobody won task j
                elif neigh.z[j] is None:
                    # agent i (self) thinks it won task j
                    if self.z[j] == self.id:
                        self.leave()
                    # agent i (self) thinks k won task j
                    elif self.z[j] == k:
                        self.update(neigh, task_id)
                    # agent i (self) thinks nobody won task j
                    elif self.z[j] is None:
                        self.leave()
                    # agent i (self) thinks some other agent m won task j
                    else:
                        m = int(self.z[j])
                        if neigh.s[m] > self.s[m]:
                            self.reset(task_id)

                # agent k thinks agent some other agent m won task j
                else:
                    m = int(neigh.z[j])
                    # agent i (self) thinks it won task j
                    if self.z[j] == self.id:
                        if (neigh.s[m] > self.s[m] and neigh.y[j] > self.y[j]) or (neigh.s[m] > self.s[m] and neigh.y[j] == self.y[j] and neigh.id < self.id):
                            self.update(neigh, task_id)
                    # agent i (self) thinks k won task j
                    elif self.z[j] == k:
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, task_id)
                        else:
                            self.reset(task_id)
                    # agent i (self) thinks m won task j
                    elif self.z[j] == m:
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, task_id)
                    # agent i (self) thinks nobody won task j
                    elif self.z[j] is None:
                        if neigh.s[m] > self.s[m]:
                            self.update(neigh, task_id)
                    else:
                        n = int(self.z[j])
                        if neigh.s[m] > self.s[m] and neigh.s[n] > self.s[n]:
                            self.update(neigh, task_id)
                        elif (neigh.s[m] > self.s[m] and neigh.y[j] > self.y[j]) or (neigh.s[m] > self.s[m] and neigh.y[j] == self.y[j] and neigh.id < self.id):
                            self.update(neigh, task_id)
                        elif neigh.s[n] > self.s[n] and self.s[m] > neigh.s[m]:
                            self.reset(task_id)
        
        self.compute_s(neigh, consensus_iter)

    def snapshot(self):
        """
        Snapshot for consesus sharing
        """
        snap = object.__new__(CBBA_Agent)
        snap.id = self.id
        snap.y = self.y[:]
        snap.z = self.z[:]
        snap.s = self.s[:]

        snap.their_net_cvg = []
        return snap