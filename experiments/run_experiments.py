"""
Experiment Runner — GCBBA + Collision Avoidance Integration
===========================================================

Changes from v1:
  - Communication range sweep: [3, 5, 8, 13, 20, 45] (meaningful connectivity levels)
  - rerun_interval is now a SWEEP parameter for dynamic configs, like task_counts / comm_ranges
    Canonical value is 50 (config_name="dynamic"), others named "dynamic_ri{X}"
  - stuck_threshold: 15
  - max_timesteps: 800 (ensures tpi=10 completes)
  - Seeds: 5 (better statistical significance)
  - FIX: Deadlock counting is now transition-based (distinct stuck-state entry events),
    not cumulative stuck-agent-timesteps. The old approach over-counted by 10-20x.
  - NEW: Tracks GCBBA trigger reason (batch completion vs interval timer)
  - NEW: Tracks total distance traveled per agent
  - NEW: Tracks path planning call count separately from allocation call count
  - Tracks whether runs hit the timestep ceiling (DNF detection)
  - Tracks initial communication graph components and diameter

IMPORTANT: This script assumes the following orchestrator fixes have been applied:
  1. Batch completion trigger in _detect_events()
  2. Reservation clearing before replanning in _plan_paths()
  3. Stuck agents no longer trigger GCBBA reruns
  See experiment_fixes.md for details.

Usage:
  python run_experiments.py --mode quick    # ~8 runs, verify pipeline
  python run_experiments.py --mode medium   # ~216 runs, initial results
  python run_experiments.py --mode full     # ~720 runs, thesis data
"""

import os
import sys
import csv
import json
import time
import threading
import argparse
import itertools
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from integration.orchestrator import IntegrationOrchestrator


# ─────────────────────────────────────────────────────────────────
#  RunMetrics dataclass — every field we record per run
# ─────────────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    # ── Identity ──
    run_id: str = ""
    config_name: str = ""
    allocation_method: str = "gcbba"
    seed: int = 0
    task_arrival_rate: float = 0.1
    queue_max_depth: int = 5
    warmup_timesteps: int = 200
    initial_tasks: int = 0
    comm_range: float = 30.0
    rerun_interval: int = 50
    stuck_threshold: int = 15
    num_agents: int = 6

    # ── Outcome ──
    makespan: int = 0
    total_steps: int = 0
    all_tasks_completed: bool = False
    hit_timestep_ceiling: bool = False
    hit_wall_clock_ceiling: bool = False   # C3: run stopped by wall-clock cap before max_timesteps
    num_tasks_total: int = 0
    num_tasks_completed: int = 0

    # ── Allocation timing ──
    num_gcbba_runs: int = 0
    total_gcbba_time_ms: float = 0.0
    avg_gcbba_time_ms: float = 0.0
    max_gcbba_time_ms: float = 0.0
    # Within-run call-to-call variability (std of individual durations).
    # Low std_gcbba_time_ms → predictable per-call cost → easier real-time guarantees.
    std_gcbba_time_ms: float = 0.0

    # ── NEW: Rerun trigger breakdown ──
    # Counts runs triggered because enough tasks completed (batch threshold crossed)
    num_gcbba_runs_batch_triggered: int = 0
    # Counts runs triggered by the periodic interval timer
    num_gcbba_runs_interval_triggered: int = 0

    # ── Safety: collisions & deadlocks ──
    num_vertex_collisions: int = 0
    num_edge_collisions: int = 0
    # FIX: num_deadlocks now counts distinct stuck-state entry events (transitions),
    # NOT cumulative stuck-agent-timesteps. An agent stuck for 20 timesteps = 1 event.
    num_deadlocks: int = 0
    # Number of allocation calls that exceeded the timeout and used the stale assignment.
    num_allocation_timeouts: int = 0

    # ── Allocation call context (A4) ──
    # Mean number of unassigned tasks present at each allocation call.
    # Contextualises timing: a slow call with 150 tasks differs from one with 10.
    avg_tasks_per_gcbba_call: float = 0.0

    # ── Agent utilization ──
    avg_idle_ratio: float = 0.0
    max_idle_ratio: float = 0.0
    # Within-run std of per-agent idle ratios (A6): high std = uneven workload distribution.
    std_idle_ratio: float = 0.0
    per_agent_tasks_completed: List[int] = field(default_factory=list)
    task_balance_std: float = 0.0

    # ── NEW: Distance traveled ──
    # Total grid steps taken across all agents (Manhattan distance, one move = 1)
    total_distance_all_agents: float = 0.0
    avg_distance_per_agent: float = 0.0
    # Per-agent breakdown saved to metrics.json only (B2)
    per_agent_distances: List[float] = field(default_factory=list)

    # ── Task duration stats ──
    avg_task_duration: float = 0.0
    max_task_duration: float = 0.0
    min_task_duration: float = 0.0
    # Timestep when the first task completed (-1 if no tasks completed) (A7)
    first_task_completion_timestep: int = -1

    # ── Communication graph topology (at t=0) ──
    initial_num_components: int = 1
    initial_diameter: int = 1

    # ── Time-series data (stored as lists, excluded from summary CSV) ──
    tasks_completed_over_time: List[int] = field(default_factory=list)
    gcbba_run_timesteps: List[int] = field(default_factory=list)
    gcbba_run_durations_ms: List[float] = field(default_factory=list)
    # A4: tasks present at each allocation call (saved to JSON only)
    gcbba_tasks_per_run: List[int] = field(default_factory=list)
    # B1: mean queue depth snapshot per timestep (saved to JSON only)
    queue_depth_over_time: List[float] = field(default_factory=list)

    # ── Energy config (from YAML) ──
    max_energy: int = 100
    charge_duration: int = 20
    charge_rate: int = 1
    charging_trigger_multiplier: float = 2.0

    # ── Energy and charging metrics ──
    # Total times any agent initiated navigation to a charging station
    num_charging_events: int = 0
    # Agent-timesteps actually spent charging (at station)
    total_charging_timesteps: int = 0
    # Agent-timesteps spent navigating toward a charging station
    total_navigating_to_charger_timesteps: int = 0
    # (charging + nav_to_charger) / total_agent_timesteps — throughput overhead
    charging_time_fraction: float = 0.0
    # Mean energy across all agents at end of simulation
    avg_final_energy: float = 0.0
    # Lowest single-agent energy at end (energy-stress indicator)
    min_final_energy: int = 0
    # Lowest energy any agent reached at ANY point during the run (A5).
    # min_final_energy can miss mid-run brownout events; this captures them.
    min_energy_ever: int = 0
    # Tasks that were dropped mid-execution because the agent had to charge
    num_tasks_aborted_for_charging: int = 0

    # ── Steady-state throughput metrics ──
    total_tasks_injected: int = 0
    tasks_dropped_by_queue_cap: int = 0
    steady_state_tasks_completed: int = 0
    throughput: float = 0.0              # tasks/timestep (post-warmup window only)
    avg_task_wait_time: float = 0.0      # injection → execution start (post-warmup)
    max_task_wait_time: float = 0.0
    avg_queue_depth: float = 0.0         # mean across stations over full run
    queue_saturation_fraction: float = 0.0  # fraction of timesteps any station at max depth

    # ── Path planning timing ──
    # Counts only non-trivial _plan_paths() calls (≥1 agent actually replanned).
    # avg/max are over those calls only; total_path_plan_time_ms sums all of them.
    num_replanning_events: int = 0
    total_path_plan_time_ms: float = 0.0
    avg_path_plan_time_ms: float = 0.0
    max_path_plan_time_ms: float = 0.0

    # ── Wall-clock timing ──
    wall_time_seconds: float = 0.0
    # Per-timestep simulation step time (measures real-time feasibility).
    # avg = mean cost of one tick; max = worst-case latency; std = consistency.
    # All three cover allocation + path planning + agent state updates per step.
    avg_step_time_ms: float = 0.0
    max_step_time_ms: float = 0.0
    std_step_time_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────
#  InstrumentedOrchestrator — wraps IntegrationOrchestrator to
#  collect metrics that the base class doesn't expose
# ─────────────────────────────────────────────────────────────────

class InstrumentedOrchestrator(IntegrationOrchestrator):
    def __init__(self, *args, **kwargs):
        # C1: pop before passing to super so IntegrationOrchestrator doesn't see it
        self._allocation_timeout_s: Optional[float] = kwargs.pop("allocation_timeout_s", None)
        # C3: per-run wall-clock cap
        self._wall_time_limit_s: Optional[float] = kwargs.pop("wall_clock_limit_s", None)
        super().__init__(*args, **kwargs)

        # C1: count allocation calls that hit the timeout
        self._num_allocation_timeouts: int = 0

        # C3: wall-clock cap state (start time set at beginning of run_simulation)
        self._run_wall_start: float = 0.0
        self._hit_wall_clock_ceiling: bool = False

        # Allocation tracking
        self._gcbba_run_count = 0
        self._gcbba_times_ms: List[float] = []
        self._gcbba_timesteps: List[int] = []

        # Task completion timeline (one entry per timestep)
        self._tasks_completed_timeline: List[int] = []

        # FIX: Deadlock tracking — count TRANSITIONS into stuck state, not persistent stuck.
        # If an agent stays stuck for 30 timesteps, that is ONE deadlock event, not 30.
        self._deadlock_count = 0
        self._previously_stuck: Set[int] = set()  # agents that were stuck last timestep

        # NEW: Trigger reason tracking
        # We need to know WHY a GCBBA rerun was triggered: batch completion or interval?
        # We detect this by checking the same logic as _detect_events() uses.
        self._gcbba_batch_triggers = 0
        self._gcbba_interval_triggers = 0
        # Snapshot of completed count at the point we DETECT the rerun, so we can
        # classify it in the same step() call where the trigger fires.
        self._completed_count_at_last_check = 0

        # A4: task count at each allocation call
        self._gcbba_tasks_per_run: List[int] = []

        # A5: minimum energy any agent ever reached during the run
        self._min_energy_ever: int = 100  # reset properly on first step

        # Path planning timing — only non-trivial calls (≥1 agent actually replanned)
        self._path_plan_times_ms: List[float] = []

        # Per-timestep wall time (covers allocation + path planning + state updates)
        self._step_times_ms: List[float] = []

        # Energy tracking
        # Transitions into is_navigating_to_charger (one per charge cycle per agent)
        self._charging_event_count: int = 0
        # Agent-timesteps spent physically at the charging station (is_charging==True)
        self._total_charging_steps: int = 0
        # Agent-timesteps spent navigating toward a charger (is_navigating_to_charger==True)
        self._total_nav_to_charger_steps: int = 0
        # Tasks dropped when an agent called start_charging() mid-execution
        self._tasks_aborted_for_charging: int = 0
        # Agent IDs that were navigating to a charger at the END of the last timestep
        # (used to detect transitions, same pattern as _previously_stuck)
        self._previously_navigating_to_charger: Set[int] = set()

    def run_allocation(self) -> None:
        """Time each allocation call and record the timestep it occurred.

        C1: For cbba/sga methods, a threading timeout prevents runaway calls from
        blocking the simulation indefinitely.  If the call exceeds the timeout, we
        log the event and continue with whatever assignment state existed before
        (i.e. the stale assignment — agents keep their current tasks).
        """
        # A4: count unassigned tasks at this moment (excludes completed + executing)
        executing = self._get_executing_task_ids()
        excluded  = self.completed_task_ids | executing
        nt_active = sum(1 for tid in self.all_task_ids if tid not in excluded)
        self._gcbba_tasks_per_run.append(nt_active)

        t0 = time.perf_counter()

        use_timeout = (
            self._allocation_timeout_s is not None
            and self.allocation_method in ("cbba", "sga")
        )
        if use_timeout:
            exc_box: List[Optional[Exception]] = [None]

            def _target():
                try:
                    IntegrationOrchestrator.run_allocation(self)
                except Exception as e:
                    exc_box[0] = e

            thread = threading.Thread(target=_target, daemon=True)
            thread.start()
            thread.join(timeout=self._allocation_timeout_s)
            if thread.is_alive():
                self._num_allocation_timeouts += 1
                print(
                    f"[t={self.current_timestep}] ALLOCATION TIMEOUT "
                    f"(>{self._allocation_timeout_s}s, {self.allocation_method.upper()}, "
                    f"{nt_active} tasks) — using stale assignment.",
                    flush=True,
                )
            elif exc_box[0] is not None:
                raise exc_box[0]
        else:
            IntegrationOrchestrator.run_allocation(self)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._gcbba_run_count += 1
        self._gcbba_times_ms.append(elapsed_ms)
        self._gcbba_timesteps.append(self.current_timestep)

    def _plan_paths(self) -> None:
        """Time path planning, but only for calls that do real work (≥1 agent replanning).

        Skips timing trivial calls (all agents have paths already) so the stats
        reflect actual A* / BFS cost rather than being diluted by thousands of
        no-op calls.  Replanning events = distinct batches of path computation.
        """
        if not any(a.needs_new_path for a in self.agent_states):
            super()._plan_paths()
            return
        t0 = time.perf_counter()
        super()._plan_paths()
        self._path_plan_times_ms.append((time.perf_counter() - t0) * 1000.0)

    def run_simulation(self, timesteps: int = 100) -> None:
        """C3: Override to add a per-run wall-clock cap.

        Replicates the base class loop and breaks early if elapsed wall time
        exceeds self._wall_time_limit_s.  Metrics are still computed correctly
        from whatever timesteps completed (throughput normalises by ss_steps).
        """
        from tqdm import tqdm
        self._run_wall_start = time.perf_counter()
        pbar = tqdm(
            range(timesteps),
            desc=f"Simulation ({self.allocation_method.upper()})",
            leave=True,
        )
        for _ in pbar:
            if self._wall_time_limit_s is not None:
                elapsed = time.perf_counter() - self._run_wall_start
                if elapsed > self._wall_time_limit_s:
                    self._hit_wall_clock_ceiling = True
                    tqdm.write(
                        f"[t={self.current_timestep}] WALL-CLOCK LIMIT "
                        f"({self._wall_time_limit_s / 60:.0f} min) exceeded "
                        f"({elapsed / 60:.1f} min elapsed) — stopping early.",
                        flush=True,
                    )
                    break
            self.step()
            done = len(self.completed_task_ids)
            q = (
                float(np.mean(list(self._induct_queue_depth.values())))
                if self._induct_queue_depth else 0
            )
            pbar.set_postfix(done=done, t=self.current_timestep, q=f"{q:.2f}", refresh=False)
            # Batch-mode early exit (mirrored from base class)
            if (
                self.completed_task_ids >= self.all_task_ids
                and self.task_arrival_rate == 0
                and self.all_task_ids
            ):
                tqdm.write(
                    f"All {len(self.completed_task_ids)} tasks completed at t={self.current_timestep}."
                )
                break

    def step(self, *args, **kwargs):
        """
        Override step() to:
          1. Track task completion timeline
          2. Detect NEW stuck agents (transitions into stuck state) for deadlock count
          3. Classify what triggered any GCBBA rerun that fires this timestep
          4. Track charging events, charging timesteps, and task abortions for energy metrics
        """
        # ── Snapshot state BEFORE this step ──────────────────────────────────
        # Count completed tasks before this step so we can detect new completions
        completed_before = len(self.completed_task_ids)

        # Energy: which agents have a task right now (so we can detect abortions)
        agents_with_task_before: Set[int] = {
            state.agent_id
            for state in self.agent_states
            if state.current_task is not None
        }
        # Energy: which agents are already navigating to a charger (to detect NEW events)
        navigating_before: Set[int] = {
            state.agent_id
            for state in self.agent_states
            if state.is_navigating_to_charger
        }

        _t0_step = time.perf_counter()
        events = super().step(*args, **kwargs)
        self._step_times_ms.append((time.perf_counter() - _t0_step) * 1000.0)

        # ── Timeline ──
        self._tasks_completed_timeline.append(len(self.completed_task_ids))

        # ── Deadlock: count new stuck events (TRANSITION detection) ──
        currently_stuck: Set[int] = {
            agent_state.agent_id
            for agent_state in self.agent_states
            if agent_state.is_stuck
        }
        # Agents that are stuck NOW but were NOT stuck last timestep = new deadlock events
        newly_stuck = currently_stuck - self._previously_stuck
        self._deadlock_count += len(newly_stuck)
        self._previously_stuck = currently_stuck

        # ── Trigger reason classification ──
        # If a GCBBA rerun fired this step, classify its cause using the same
        # batch_threshold logic as _detect_events(). We read the batch_threshold
        # from the orchestrator's own parameters.
        if events.gcbba_rerun:
            batch_threshold = max(2, self.num_agents // 3)
            completed_since_last = (
                len(self.completed_task_ids) - self._completed_at_last_gcbba
            )
            if completed_since_last >= batch_threshold:
                self._gcbba_batch_triggers += 1
            else:
                self._gcbba_interval_triggers += 1

        # ── Energy tracking ──────────────────────────────────────────────────
        # A5: track global minimum energy across all agents and all timesteps
        step_min_energy = min(state.energy for state in self.agent_states)
        if step_min_energy < self._min_energy_ever:
            self._min_energy_ever = step_min_energy

        for state in self.agent_states:
            aid = state.agent_id
            was_navigating = aid in navigating_before

            # New charging event: agent just started navigating to a charger this step
            if state.is_navigating_to_charger and not was_navigating:
                self._charging_event_count += 1
                # If the agent had an active task before this step, it was aborted
                if aid in agents_with_task_before:
                    self._tasks_aborted_for_charging += 1

            # Accumulate per-agent-timestep counts in energy-related states
            if state.is_navigating_to_charger:
                self._total_nav_to_charger_steps += 1
            if state.is_charging:
                self._total_charging_steps += 1

        return events

    # ── Graph info at simulation start ──────────────────────────────

    def get_initial_graph_info(self) -> Tuple[int, int]:
        """
        Reconstruct the communication graph from agent positions at t=0
        and return (num_connected_components, diameter).
        diameter = -1 if graph is disconnected.
        """
        import networkx as nx
        from gcbba.tools_warehouse import create_graph_with_range

        positions = []
        for agent_state in self.agent_states:
            pos = agent_state.get_position()
            continuous_pos = self.grid_map.grid_to_continuous(*pos)
            positions.append((
                continuous_pos[0], continuous_pos[1], continuous_pos[2],
                agent_state.agent_id
            ))

        raw_graph, _ = create_graph_with_range(positions, self.comm_range)
        num_components = nx.number_connected_components(raw_graph)
        diameter = (
            nx.diameter(raw_graph) if nx.is_connected(raw_graph) else -1
        )
        return num_components, diameter

    # ── Post-hoc collision check ─────────────────────────────────────

    def _check_collisions(self) -> Tuple[int, int]:
        """
        Walk position histories for all agents and count:
          - vertex collisions: two agents at same (x,y,z,t)
          - edge collisions: two agents swap positions in one timestep
        """
        position_map: Dict[Tuple, int] = {}
        vertex_collisions = 0
        edge_collisions = 0
        agent_trajectories: Dict[int, Dict[int, Tuple]] = {}

        for agent_state in self.agent_states:
            aid = agent_state.agent_id
            agent_trajectories[aid] = {}
            for (x, y, z, t) in agent_state.position_history:
                key = (x, y, z, t)
                agent_trajectories[aid][t] = (x, y, z)
                if key in position_map and position_map[key] != aid:
                    vertex_collisions += 1
                else:
                    position_map[key] = aid

        agent_ids = list(agent_trajectories.keys())
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a1, a2 = agent_ids[i], agent_ids[j]
                traj1, traj2 = agent_trajectories[a1], agent_trajectories[a2]
                for t in set(traj1.keys()) & set(traj2.keys()):
                    if t - 1 not in traj1 or t - 1 not in traj2:
                        continue
                    if (
                        traj1[t - 1] == traj2[t]
                        and traj2[t - 1] == traj1[t]
                        and traj1[t - 1] != traj1[t]
                    ):
                        edge_collisions += 1

        return vertex_collisions, edge_collisions

    # ── Distance computation ─────────────────────────────────────────

    def _compute_distances(self) -> Tuple[float, float]:
        """
        Compute total grid-steps traveled per agent (Manhattan distance, 1 cell = 1 step).
        Returns (total_all_agents, avg_per_agent).
        """
        distances = []
        for agent_state in self.agent_states:
            hist = agent_state.position_history
            dist = 0
            for k in range(1, len(hist)):
                prev = hist[k - 1][:3]
                curr = hist[k][:3]
                if prev != curr:
                    dist += (
                        abs(curr[0] - prev[0])
                        + abs(curr[1] - prev[1])
                        + abs(curr[2] - prev[2])
                    )
            distances.append(dist)
        total = sum(distances)
        avg = float(np.mean(distances)) if distances else 0.0
        return round(total, 2), round(avg, 2)

    # ── Main metrics collection ──────────────────────────────────────

    def collect_metrics(
        self,
        config_name: str,
        seed: int,
        task_arrival_rate: float,
        queue_max_depth: int,
        warmup_timesteps: int,
        comm_range: float,
        rerun_interval: int,
        stuck_threshold: int,
        wall_time: float,
        max_timesteps: int,
    ) -> RunMetrics:
        m = RunMetrics()

        # Identity
        m.config_name = config_name
        m.allocation_method = self.allocation_method
        m.seed = seed
        m.task_arrival_rate = task_arrival_rate
        m.queue_max_depth = queue_max_depth
        m.warmup_timesteps = warmup_timesteps
        m.initial_tasks = self.initial_tasks
        m.comm_range = comm_range
        m.rerun_interval = rerun_interval
        m.stuck_threshold = stuck_threshold
        m.num_agents = self.num_agents
        m.wall_time_seconds = round(wall_time, 3)

        # Outcome
        m.total_steps = self.current_timestep
        m.num_tasks_total = len(self.all_task_ids)
        m.num_tasks_completed = len(self.completed_task_ids)
        m.all_tasks_completed = (self.completed_task_ids == self.all_task_ids)
        m.makespan = self.current_timestep if m.all_tasks_completed else -1
        m.hit_timestep_ceiling = (
            not m.all_tasks_completed
            and self.current_timestep >= max_timesteps - 1
        )

        # Allocation timing
        m.num_gcbba_runs = self._gcbba_run_count
        m.total_gcbba_time_ms = round(sum(self._gcbba_times_ms), 2)
        m.avg_gcbba_time_ms = (
            round(np.mean(self._gcbba_times_ms), 2) if self._gcbba_times_ms else 0
        )
        m.max_gcbba_time_ms = (
            round(max(self._gcbba_times_ms), 2) if self._gcbba_times_ms else 0
        )
        m.std_gcbba_time_ms = (
            round(float(np.std(self._gcbba_times_ms)), 2) if self._gcbba_times_ms else 0
        )
        m.gcbba_run_timesteps = self._gcbba_timesteps
        m.gcbba_run_durations_ms = [round(t, 2) for t in self._gcbba_times_ms]

        # Trigger reason breakdown
        m.num_gcbba_runs_batch_triggered = self._gcbba_batch_triggers
        m.num_gcbba_runs_interval_triggered = self._gcbba_interval_triggers

        # Safety
        vertex_col, edge_col = self._check_collisions()
        m.num_vertex_collisions = vertex_col
        m.num_edge_collisions = edge_col
        m.num_deadlocks = self._deadlock_count  # transition-based, see step()

        # Communication graph
        num_comp, diam = self.get_initial_graph_info()
        m.initial_num_components = num_comp
        m.initial_diameter = diam

        # Agent utilization
        idle_ratios, tasks_per_agent = [], []
        for agent_state in self.agent_states:
            total_steps = max(len(agent_state.position_history), 1)
            idle_steps = sum(
                1
                for k in range(1, len(agent_state.position_history))
                if agent_state.position_history[k - 1][:3]
                == agent_state.position_history[k][:3]
            )
            idle_ratios.append(idle_steps / total_steps)
            tasks_per_agent.append(len(agent_state.completed_tasks))

        m.avg_idle_ratio = round(float(np.mean(idle_ratios)), 4)
        m.max_idle_ratio = round(float(max(idle_ratios)), 4)
        m.std_idle_ratio = round(float(np.std(idle_ratios)), 4)   # A6
        m.per_agent_tasks_completed = tasks_per_agent
        m.task_balance_std = round(float(np.std(tasks_per_agent)), 3)

        # Distance traveled — also save per-agent breakdown (B2)
        agent_distances = []
        for agent_state in self.agent_states:
            hist = agent_state.position_history
            dist = sum(
                abs(hist[k][0] - hist[k-1][0])
                + abs(hist[k][1] - hist[k-1][1])
                + abs(hist[k][2] - hist[k-1][2])
                for k in range(1, len(hist))
                if hist[k][:3] != hist[k-1][:3]
            )
            agent_distances.append(round(float(dist), 2))
        m.total_distance_all_agents = round(sum(agent_distances), 2)
        m.avg_distance_per_agent    = round(float(np.mean(agent_distances)) if agent_distances else 0.0, 2)
        m.per_agent_distances       = agent_distances                        # B2

        # Task duration stats
        task_durations = []
        for agent_state in self.agent_states:
            for task in agent_state.completed_tasks:
                if task.start_time is not None and task.completion_time is not None:
                    task_durations.append(task.completion_time - task.start_time)
        if task_durations:
            m.avg_task_duration = round(float(np.mean(task_durations)), 2)
            m.max_task_duration = float(max(task_durations))
            m.min_task_duration = float(min(task_durations))

        # A4: allocation call context
        m.gcbba_tasks_per_run = self._gcbba_tasks_per_run
        if self._gcbba_tasks_per_run:
            m.avg_tasks_per_gcbba_call = round(float(np.mean(self._gcbba_tasks_per_run)), 2)

        # A7: first task completion timestep
        for t_idx, n_done in enumerate(self._tasks_completed_timeline):
            if n_done > 0:
                m.first_task_completion_timestep = t_idx
                break

        # Time-series
        m.tasks_completed_over_time = self._tasks_completed_timeline
        m.queue_depth_over_time     = list(self._queue_depth_snapshots)  # B1

        # Energy config
        m.max_energy = self.max_energy
        m.charge_duration = self.charge_duration
        m.charge_rate = self.charge_rate
        m.charging_trigger_multiplier = self.charging_trigger_multiplier

        # Energy metrics
        m.num_charging_events = self._charging_event_count
        m.total_charging_timesteps = self._total_charging_steps
        m.total_navigating_to_charger_timesteps = self._total_nav_to_charger_steps
        total_agent_timesteps = max(self.num_agents * self.current_timestep, 1)
        m.charging_time_fraction = round(
            (self._total_charging_steps + self._total_nav_to_charger_steps)
            / total_agent_timesteps,
            4,
        )
        final_energies = [state.energy for state in self.agent_states]
        m.avg_final_energy = round(float(np.mean(final_energies)), 2)
        m.min_final_energy = int(min(final_energies))
        m.min_energy_ever  = int(self._min_energy_ever)              # A5
        m.num_tasks_aborted_for_charging = self._tasks_aborted_for_charging

        # Steady-state throughput metrics
        m.total_tasks_injected = self._next_task_id
        m.tasks_dropped_by_queue_cap = self._tasks_dropped_by_cap

        # Deduplicate by task_id: GCBBA with disconnected comm graphs can assign the same
        # task to multiple agents (each subgraph independently picks a winner), so the same
        # task_id may appear in several agents' completed_tasks lists.  We take the first
        # occurrence (lowest start_time) to correctly represent when the task was first done.
        seen_ss_ids: set = set()
        ss_tasks = []
        for agent_state in self.agent_states:
            for task in agent_state.completed_tasks:
                if (
                    task.start_time is not None
                    and task.start_time >= warmup_timesteps
                    and task.task_id not in seen_ss_ids
                ):
                    seen_ss_ids.add(task.task_id)
                    ss_tasks.append(task)
        ss_steps = max(1, self.current_timestep - warmup_timesteps)
        m.steady_state_tasks_completed = len(ss_tasks)
        m.throughput = round(len(ss_tasks) / ss_steps, 4)

        wait_times = []
        for task in ss_tasks:
            inj_t = self._task_injection_time.get(task.task_id)
            if inj_t is not None and task.start_time is not None:
                wait_times.append(task.start_time - inj_t)
        if wait_times:
            m.avg_task_wait_time = round(float(np.mean(wait_times)), 2)
            m.max_task_wait_time = float(max(wait_times))

        if self._queue_depth_snapshots:
            m.avg_queue_depth = round(float(np.mean(self._queue_depth_snapshots)), 3)
            sat_count = sum(
                1 for s in self._queue_depth_snapshots if s >= self.induct_queue_capacity
            )
            m.queue_saturation_fraction = round(
                sat_count / len(self._queue_depth_snapshots), 4
            )

        # Path planning timing
        m.num_replanning_events = len(self._path_plan_times_ms)
        if self._path_plan_times_ms:
            m.total_path_plan_time_ms = round(sum(self._path_plan_times_ms), 2)
            m.avg_path_plan_time_ms   = round(float(np.mean(self._path_plan_times_ms)), 2)
            m.max_path_plan_time_ms   = round(float(max(self._path_plan_times_ms)), 2)

        # Per-timestep step time
        if self._step_times_ms:
            m.avg_step_time_ms = round(float(np.mean(self._step_times_ms)), 3)
            m.max_step_time_ms = round(float(max(self._step_times_ms)), 3)
            m.std_step_time_ms = round(float(np.std(self._step_times_ms)), 3)

        # C1: allocation timeout count
        m.num_allocation_timeouts = self._num_allocation_timeouts

        # C3: wall-clock ceiling flag
        m.hit_wall_clock_ceiling = self._hit_wall_clock_ceiling

        # Build unique run ID — batch mode uses "it{N}" tag, steady-state uses "ar{rate}"
        if self.initial_tasks > 0 and task_arrival_rate == 0:
            m.run_id = f"{config_name}_it{self.initial_tasks}_cr{int(comm_range)}_s{seed}"
        else:
            m.run_id = f"{config_name}_ar{task_arrival_rate}_cr{int(comm_range)}_s{seed}"
        return m


# ─────────────────────────────────────────────────────────────────
#  Experiment config generator
# ─────────────────────────────────────────────────────────────────

def get_experiment_configs(mode: str = "full") -> List[Dict]:
    """
    Build the list of experimental configurations to run.

    Parameters swept:
      - arrival_rates  : tasks per timestep per induct station
      - comm_ranges    : communication range (grid units)
      - rerun_interval : ONLY for dynamic GCBBA (static/cbba/sga use 999999)

    Config names:
      "static"         → GCBBA, responds only to idle+arrival, no proactive replanning
      "dynamic"        → GCBBA, rerun_interval = 50 (canonical replanning value)
      "dynamic_ri{X}"  → GCBBA, rerun_interval = X  (sensitivity sweep)
      "cbba"           → standard CBBA baseline
      "sga"            → centralized SGA baseline

    Arrival rate rationale (8 stations, ~6 agents, ~20 ts/task):
      Capacity ≈ 0.3 tasks/ts total → per-station ≈ 0.037 tasks/ts/station.
      0.01 → ~27% capacity (light)  |  0.03 → ~81% (near-knee)
      0.05 → ~135% (slight overload)|  0.1  → ~270% (heavy)
      0.15 → ~400%                   |  0.2  → ~540% (extreme)
      Sweep captures light load, the saturation knee (~0.037), and severe overload.
    """
    configs = []

    if mode == "quick":
        seeds = [42]
        arrival_rates = [0.05]
        comm_ranges = [13, 45]
        rerun_intervals = [50]
        batch_task_counts = [80]

    elif mode == "medium":
        seeds = [42, 123, 456]
        arrival_rates = [0.01, 0.03, 0.05, 0.1]
        comm_ranges = [5, 8, 13, 45]
        rerun_intervals = [25, 50, 100]
        batch_task_counts = [40, 80, 160]

    else:  # full
        seeds = [42, 123, 456, 789, 1024]
        arrival_rates = [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]
        comm_ranges = [3, 5, 8, 13, 20, 45]
        rerun_intervals = [10, 25, 50, 100, 200]
        batch_task_counts = [20, 40, 80, 160]

    STUCK_THRESHOLD = 15
    MAX_TIMESTEPS = 1500
    WARMUP_TIMESTEPS = 300   # increased from 200: more time for charging + queue to stabilise
    QUEUE_MAX_DEPTH = 10     # increased from 5: more realistic physical buffer per induct station
    BATCH_MAX_TIMESTEPS = 3000  # Batch mode needs more headroom to complete all tasks
    # Pre-load steady-state runs with tasks so agents are active from t=0 and the
    # warmup window captures loaded behaviour rather than empty-queue ramp-up.
    # 2 × num_agents (6) = 12 gives every agent an immediate assignment without
    # flooding the queues (capacity = QUEUE_MAX_DEPTH × 8 stations = 80 tasks).
    SS_INITIAL_TASKS = 12
    # C2: CBBA/SGA are much slower per-call at high loads; cap timesteps to prevent
    # multi-hour runs.  At ar >= 0.1 the simulation is overloaded anyway.
    # With WARMUP_TIMESTEPS=300 a cap of 800 gives 500 ts of steady-state window.
    CBBA_SGA_SS_CAPPED_TIMESTEPS = 800
    ALLOCATION_TIMEOUT_S = 10.0  # C1: hard cap on a single allocation call (reduced from 30s)
    WALL_CLOCK_LIMIT_S = 600.0   # C3: hard cap on total run wall time (10 min)

    for ar, cr in itertools.product(arrival_rates, comm_ranges):

        # ── Static GCBBA (only reallocates on idle+arrival, no proactive replan) ──
        configs.append({
            "config_name": "static",
            "allocation_method": "gcbba",
            "task_arrival_rate": ar,
            "initial_tasks": SS_INITIAL_TASKS,
            "queue_max_depth": QUEUE_MAX_DEPTH,
            "warmup_timesteps": WARMUP_TIMESTEPS,
            "comm_range": cr,
            "rerun_interval": 999999,   # effectively infinite → no periodic reruns
            "stuck_threshold": STUCK_THRESHOLD,
            "seeds": seeds,
            "max_timesteps": MAX_TIMESTEPS,
        })

        # ── Dynamic GCBBA — sweep over all rerun intervals ──────────
        for ri in rerun_intervals:
            name = "dynamic" if ri == 50 else f"dynamic_ri{ri}"
            configs.append({
                "config_name": name,
                "allocation_method": "gcbba",
                "task_arrival_rate": ar,
                "initial_tasks": SS_INITIAL_TASKS,
                "queue_max_depth": QUEUE_MAX_DEPTH,
                "warmup_timesteps": WARMUP_TIMESTEPS,
                "comm_range": cr,
                "rerun_interval": ri,
                "stuck_threshold": STUCK_THRESHOLD,
                "seeds": seeds,
                "max_timesteps": MAX_TIMESTEPS,
            })

        # ── Standard CBBA baseline ───────────────────────────────────
        # C2: cap timesteps at high arrival rates to prevent multi-hour runs
        cbba_sga_max_ts = CBBA_SGA_SS_CAPPED_TIMESTEPS if ar >= 0.1 else MAX_TIMESTEPS
        configs.append({
            "config_name": "cbba",
            "allocation_method": "cbba",
            "task_arrival_rate": ar,
            "initial_tasks": SS_INITIAL_TASKS,
            "queue_max_depth": QUEUE_MAX_DEPTH,
            "warmup_timesteps": WARMUP_TIMESTEPS,
            "comm_range": cr,
            "rerun_interval": 999999,
            "stuck_threshold": STUCK_THRESHOLD,
            "seeds": seeds,
            "max_timesteps": cbba_sga_max_ts,
            "allocation_timeout_s": ALLOCATION_TIMEOUT_S,  # C1
            "wall_clock_limit_s": WALL_CLOCK_LIMIT_S,      # C3
        })

        # ── SGA centralized baseline ─────────────────────────────────
        configs.append({
            "config_name": "sga",
            "allocation_method": "sga",
            "task_arrival_rate": ar,
            "initial_tasks": SS_INITIAL_TASKS,
            "queue_max_depth": QUEUE_MAX_DEPTH,
            "warmup_timesteps": WARMUP_TIMESTEPS,
            "comm_range": cr,
            "rerun_interval": 999999,
            "stuck_threshold": STUCK_THRESHOLD,
            "seeds": seeds,
            "max_timesteps": cbba_sga_max_ts,
            "allocation_timeout_s": ALLOCATION_TIMEOUT_S,  # C1
            "wall_clock_limit_s": WALL_CLOCK_LIMIT_S,      # C3
        })

    # ── Batch mode configs (initial_tasks > 0, rate = 0) ─────────────────
    for n_tasks, cr in itertools.product(batch_task_counts, comm_ranges):

        # Static GCBBA batch
        configs.append({
            "config_name": "static_batch",
            "allocation_method": "gcbba",
            "task_arrival_rate": 0,
            "initial_tasks": n_tasks,
            "queue_max_depth": QUEUE_MAX_DEPTH,
            "warmup_timesteps": 0,
            "comm_range": cr,
            "rerun_interval": 999999,
            "stuck_threshold": STUCK_THRESHOLD,
            "seeds": seeds,
            "max_timesteps": BATCH_MAX_TIMESTEPS,
        })

        # Dynamic GCBBA batch (canonical ri=50)
        configs.append({
            "config_name": "dynamic_batch",
            "allocation_method": "gcbba",
            "task_arrival_rate": 0,
            "initial_tasks": n_tasks,
            "queue_max_depth": QUEUE_MAX_DEPTH,
            "warmup_timesteps": 0,
            "comm_range": cr,
            "rerun_interval": 50,
            "stuck_threshold": STUCK_THRESHOLD,
            "seeds": seeds,
            "max_timesteps": BATCH_MAX_TIMESTEPS,
        })

        # CBBA batch baseline
        configs.append({
            "config_name": "cbba_batch",
            "allocation_method": "cbba",
            "task_arrival_rate": 0,
            "initial_tasks": n_tasks,
            "queue_max_depth": QUEUE_MAX_DEPTH,
            "warmup_timesteps": 0,
            "comm_range": cr,
            "rerun_interval": 999999,
            "stuck_threshold": STUCK_THRESHOLD,
            "seeds": seeds,
            "max_timesteps": BATCH_MAX_TIMESTEPS,
            "allocation_timeout_s": ALLOCATION_TIMEOUT_S,  # C1
            "wall_clock_limit_s": WALL_CLOCK_LIMIT_S,      # C3
        })

        # SGA batch baseline
        configs.append({
            "config_name": "sga_batch",
            "allocation_method": "sga",
            "task_arrival_rate": 0,
            "initial_tasks": n_tasks,
            "queue_max_depth": QUEUE_MAX_DEPTH,
            "warmup_timesteps": 0,
            "comm_range": cr,
            "rerun_interval": 999999,
            "stuck_threshold": STUCK_THRESHOLD,
            "seeds": seeds,
            "max_timesteps": BATCH_MAX_TIMESTEPS,
            "allocation_timeout_s": ALLOCATION_TIMEOUT_S,  # C1
            "wall_clock_limit_s": WALL_CLOCK_LIMIT_S,      # C3
        })

    return configs


# ─────────────────────────────────────────────────────────────────
#  Single experiment runner
# ─────────────────────────────────────────────────────────────────

def run_single_experiment(
    config_path: str,
    config_name: str,
    task_arrival_rate: float,
    queue_max_depth: int,
    warmup_timesteps: int,
    comm_range: float,
    rerun_interval: int,
    stuck_threshold: int,
    seed: int,
    max_timesteps: int,
    allocation_method: str = "gcbba",
    initial_tasks: int = 0,
    allocation_timeout_s: Optional[float] = None,
    wall_clock_limit_s: Optional[float] = None,
) -> Tuple[RunMetrics, "InstrumentedOrchestrator"]:
    np.random.seed(seed)

    orch = InstrumentedOrchestrator(
        config_path=config_path,
        task_arrival_rate=task_arrival_rate,
        induct_queue_capacity=queue_max_depth,
        warmup_timesteps=warmup_timesteps,
        initial_tasks=initial_tasks,
        comm_range=comm_range,
        rerun_interval=rerun_interval,
        stuck_threshold=stuck_threshold,
        max_plan_time=200,   # A* search horizon — 30×30 grid, 200 is plenty
        allocation_method=allocation_method,
        allocation_timeout_s=allocation_timeout_s,
        wall_clock_limit_s=wall_clock_limit_s,
    )

    t0 = time.perf_counter()
    orch.run_simulation(timesteps=max_timesteps)
    wall_time = time.perf_counter() - t0

    metrics = orch.collect_metrics(
        config_name, seed, task_arrival_rate, queue_max_depth,
        warmup_timesteps, comm_range, rerun_interval, stuck_threshold,
        wall_time, max_timesteps,
    )
    return metrics, orch


# ─────────────────────────────────────────────────────────────────
#  Output helpers
# ─────────────────────────────────────────────────────────────────

def save_run_results(
    metrics: RunMetrics,
    orch: "InstrumentedOrchestrator",
    output_dir: str,
) -> None:
    """Save per-run metrics JSON and agent trajectory CSV."""
    run_dir = os.path.join(output_dir, metrics.run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(asdict(metrics), f, indent=2, default=str)

    traj_path = os.path.join(run_dir, "trajectories.csv")
    with open(traj_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "x", "y", "z", "timestep"])
        for agent_state in orch.agent_states:
            for (x, y, z, t) in agent_state.position_history:
                writer.writerow([agent_state.agent_id, x, y, z, t])


# Fields written to summary.csv (list-type fields are excluded — they live in metrics.json)
SUMMARY_FIELDS = [
    "run_id", "config_name", "allocation_method", "seed",
    "task_arrival_rate", "initial_tasks", "queue_max_depth", "warmup_timesteps",
    "comm_range", "rerun_interval", "stuck_threshold", "num_agents",
    "num_tasks_total", "num_tasks_completed",
    "all_tasks_completed", "hit_timestep_ceiling", "hit_wall_clock_ceiling", "makespan", "total_steps",
    "num_gcbba_runs", "avg_tasks_per_gcbba_call",
    "num_gcbba_runs_batch_triggered", "num_gcbba_runs_interval_triggered",
    "total_gcbba_time_ms", "avg_gcbba_time_ms", "max_gcbba_time_ms", "std_gcbba_time_ms",
    "num_vertex_collisions", "num_edge_collisions", "num_deadlocks", "num_allocation_timeouts",
    "avg_idle_ratio", "max_idle_ratio", "std_idle_ratio", "task_balance_std",
    "total_distance_all_agents", "avg_distance_per_agent",
    "avg_task_duration", "max_task_duration", "min_task_duration",
    "first_task_completion_timestep",
    "initial_num_components", "initial_diameter",
    "max_energy", "charge_duration", "charge_rate", "charging_trigger_multiplier",
    "num_charging_events", "total_charging_timesteps",
    "total_navigating_to_charger_timesteps", "charging_time_fraction",
    "avg_final_energy", "min_final_energy", "min_energy_ever", "num_tasks_aborted_for_charging",
    "total_tasks_injected", "tasks_dropped_by_queue_cap",
    "steady_state_tasks_completed", "throughput",
    "avg_task_wait_time", "max_task_wait_time",
    "avg_queue_depth", "queue_saturation_fraction",
    "num_replanning_events",
    "total_path_plan_time_ms", "avg_path_plan_time_ms", "max_path_plan_time_ms",
    "wall_time_seconds",
    "avg_step_time_ms", "max_step_time_ms", "std_step_time_ms",
]


def save_summary_csv(all_metrics: List[RunMetrics], output_dir: str) -> None:
    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: getattr(m, k) for k in SUMMARY_FIELDS})

    total = len(all_metrics)
    completed = sum(1 for m in all_metrics if m.all_tasks_completed)
    ceiling = sum(1 for m in all_metrics if m.hit_timestep_ceiling)
    cols = sum(m.num_vertex_collisions + m.num_edge_collisions for m in all_metrics)
    deads = sum(m.num_deadlocks for m in all_metrics)

    print(f"\n{'='*70}")
    print(f"Summary CSV: {summary_path}")
    print(
        f"Runs: {total} | Completed: {completed} | Ceiling: {ceiling} | "
        f"Collisions: {cols} | Deadlocks (distinct events): {deads}"
    )
    print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────
#  Optimality ratio computation (runs automatically after experiments)
# ─────────────────────────────────────────────────────────────────

def compute_and_save_optimality_ratios(output_dir: str) -> None:
    """
    Read summary.csv, compute throughput ratio relative to SGA for every run,
    and write summary_with_optimality.csv alongside it.

    ratio = method_throughput / sga_throughput

    Grouped by (seed, task_arrival_rate, comm_range) so each method is compared
    to SGA under the same arrival load and connectivity.

    Columns added:
      sga_reference_throughput — SGA throughput for this (seed, ar, cr) group
      throughput_ratio         — method_throughput / sga_reference_throughput
      throughput_ratio_valid   — True when both method and SGA have throughput > 0
    """
    import pandas as pd

    summary_path = os.path.join(output_dir, "summary.csv")
    if not os.path.exists(summary_path):
        print("  [!] Skipping optimality computation (summary.csv not found)")
        return

    df = pd.read_csv(summary_path)

    df["sga_reference_throughput"] = float("nan")
    df["throughput_ratio"] = float("nan")
    df["throughput_ratio_valid"] = False

    for _, group_df in df.groupby(["seed", "task_arrival_rate", "comm_range"]):
        sga_rows = group_df[
            (group_df["config_name"] == "sga")
            & (group_df["throughput"] > 0)
        ]
        if sga_rows.empty:
            continue

        sga_tput = float(sga_rows["throughput"].iloc[0])
        if sga_tput <= 0:
            continue

        df.loc[group_df.index, "sga_reference_throughput"] = sga_tput

        for idx in group_df.index:
            method_tput = float(df.loc[idx, "throughput"])
            if method_tput > 0:
                df.loc[idx, "throughput_ratio_valid"] = True
                df.loc[idx, "throughput_ratio"] = round(method_tput / sga_tput, 4)

    # ── Console summary ──
    valid = df[df["throughput_ratio_valid"] == True]
    print(f"\n{'='*70}")
    print("THROUGHPUT RATIO SUMMARY  (method_throughput / sga_throughput)")
    print("Ratio = 1.0 -> matches SGA.  >1.0 -> better than SGA.")
    print(f"{'='*70}")

    if valid.empty:
        print("  WARNING: No valid ratios - check that runs produced throughput > 0.")
    else:
        for method in ["static", "dynamic", "cbba"]:
            m_df = valid[valid["config_name"] == method]
            if m_df.empty:
                continue
            print(
                f"  {method.upper():20s} "
                f"mean={m_df['throughput_ratio'].mean():.3f}  "
                f"std={m_df['throughput_ratio'].std():.3f}  "
                f"min={m_df['throughput_ratio'].min():.3f}  "
                f"max={m_df['throughput_ratio'].max():.3f}  "
                f"n={len(m_df)}"
            )
        ri_methods = sorted(
            m for m in valid["config_name"].unique()
            if m.startswith("dynamic_ri")
        )
        if ri_methods:
            print("  --- Rerun interval sensitivity ---")
            for method in ri_methods:
                m_df = valid[valid["config_name"] == method]
                ri_val = method.replace("dynamic_ri", "")
                print(
                    f"  {'ri='+ri_val:20s} "
                    f"mean={m_df['throughput_ratio'].mean():.3f}  "
                    f"n={len(m_df)}"
                )

    print(f"{'='*70}\n")

    # ── Save enriched CSV ──
    out_path = os.path.join(output_dir, "summary_with_optimality.csv")
    df.to_csv(out_path, index=False)
    print(f"  Throughput ratios saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GCBBA experiments")
    parser.add_argument(
        "--config",
        choices=[
            "all", "ss_only", "batch_only",
            "static_only", "dynamic_only", "cbba_only",
            "sga_only", "baselines_only", "sensitivity_only",
        ],
        default="all",
        help=(
            "Which subset of configs to run. "
            "'ss_only' = steady-state configs only (task_arrival_rate > 0). "
            "'batch_only' = batch configs only (initial_tasks > 0, rate = 0). "
            "'sensitivity_only' runs only dynamic_ri* sweep configs."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "medium", "full"],
        default="medium",
    )
    parser.add_argument("--output", default=None, help="Override output directory")
    args = parser.parse_args()

    config_path = os.path.join(
        PROJECT_ROOT, "config", "gridworld_warehouse_small.yaml"
    )
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or os.path.join(
        PROJECT_ROOT, "results", "experiments", timestamp
    )
    os.makedirs(output_dir, exist_ok=True)

    all_configs = get_experiment_configs(args.mode)

    # ── Filter configs based on --config flag ──
    if args.config == "ss_only":
        configs = [c for c in all_configs if c["task_arrival_rate"] > 0]
    elif args.config == "batch_only":
        configs = [c for c in all_configs if c.get("initial_tasks", 0) > 0]
    elif args.config == "static_only":
        configs = [c for c in all_configs if c["config_name"] == "static"]
    elif args.config == "dynamic_only":
        # includes both "dynamic" (canonical) and "dynamic_ri*" (sweep)
        configs = [
            c for c in all_configs
            if c["config_name"] == "dynamic"
            or c["config_name"].startswith("dynamic_ri")
        ]
    elif args.config == "cbba_only":
        configs = [c for c in all_configs if c["config_name"] == "cbba"]
    elif args.config == "sga_only":
        configs = [c for c in all_configs if c["config_name"] == "sga"]
    elif args.config == "baselines_only":
        configs = [
            c for c in all_configs if c["config_name"] in {"cbba", "sga"}
        ]
    elif args.config == "sensitivity_only":
        # Only the rerun-interval sweep configs (excludes canonical "dynamic")
        configs = [
            c for c in all_configs
            if c["config_name"].startswith("dynamic_ri")
        ]
    else:
        configs = all_configs

    total_runs = sum(len(c["seeds"]) for c in configs)

    print(f"\n{'='*70}")
    print(f"GCBBA Experiments | mode={args.mode} | {total_runs} total runs")
    print(f"Arrival rates:   {sorted(set(c['task_arrival_rate'] for c in configs))}")
    print(f"Comm ranges:     {sorted(set(c['comm_range'] for c in configs))}")
    print(
        f"Rerun intervals: "
        f"{sorted(set(c['rerun_interval'] for c in configs if c['rerun_interval'] < 999999))}"
    )
    print(f"Output:          {output_dir}")
    print(f"{'='*70}\n")

    # Save experiment metadata
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "config_filter": args.config,
                "timestamp": timestamp,
                "total_runs": total_runs,
                "configs": configs,
            },
            f,
            indent=2,
        )

    all_metrics: List[RunMetrics] = []
    run_num = 0

    for cfg in configs:
        for seed in cfg["seeds"]:
            run_num += 1
            ri_str = (
                f"ri={cfg['rerun_interval']}"
                if cfg["rerun_interval"] < 999999
                else "ri=static"
            )
            label = (
                f"[{run_num}/{total_runs}] "
                f"{cfg['config_name']} "
                f"ar={cfg['task_arrival_rate']} "
                f"cr={cfg['comm_range']} "
                f"{ri_str} "
                f"s={seed}"
            )
            print(f"\n  {label}")

            try:
                metrics, orch = run_single_experiment(
                    config_path,
                    cfg["config_name"],
                    cfg["task_arrival_rate"],
                    cfg["queue_max_depth"],
                    cfg["warmup_timesteps"],
                    cfg["comm_range"],
                    cfg["rerun_interval"],
                    cfg["stuck_threshold"],
                    seed,
                    cfg["max_timesteps"],
                    allocation_method=cfg.get("allocation_method", "gcbba"),
                    initial_tasks=cfg.get("initial_tasks", 0),
                    allocation_timeout_s=cfg.get("allocation_timeout_s", None),
                    wall_clock_limit_s=cfg.get("wall_clock_limit_s", None),
                )

                save_run_results(metrics, orch, output_dir)
                all_metrics.append(metrics)

                trigger_str = (
                    f"batch={metrics.num_gcbba_runs_batch_triggered}"
                    f"/interval={metrics.num_gcbba_runs_interval_triggered}"
                    if cfg["rerun_interval"] < 999999
                    else ""
                )
                print(
                    f"    tput={metrics.throughput:.4f}t/ts "
                    f"wait={metrics.avg_task_wait_time:.1f}ts "
                    f"q={metrics.avg_queue_depth:.2f} "
                    f"dropped={metrics.tasks_dropped_by_queue_cap} "
                    f"alloc={metrics.num_gcbba_runs} {trigger_str} "
                    f"dead={metrics.num_deadlocks} "
                    f"col={metrics.num_vertex_collisions}v "
                    f"chrg={metrics.num_charging_events}evt "
                    f"graph={metrics.initial_num_components}comp "
                    f"wall={metrics.wall_time_seconds:.1f}s"
                )

            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()

    if all_metrics:
        save_summary_csv(all_metrics, output_dir)
        compute_and_save_optimality_ratios(output_dir)

    print(f"\nDone. Results in: {output_dir}")


if __name__ == "__main__":
    main()