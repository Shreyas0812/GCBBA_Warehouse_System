"""
run_experiments.py

Experiment runner that sweeps across:
- Allocation methods: ["gcbba", "sga", "cbba"]
- Communication ranges: [3, 5, 8, 13, 20, 45]
- Task loads: [2, 5, 10] tasks per induct station
- Seeds: [0, 1, 2, 3, 4]

Drop-in replacement for run_experiments.py — adds allocation_method as 
an additional sweep dimension. Uses the modified IntegrationOrchestrator
from orchestrator.py.

Results are saved per-run as CSV with allocation_method as a column.
"""

import os
import csv
import time
import traceback
import numpy as np
from itertools import product
from datetime import datetime

from integration.orchestrator import IntegrationOrchestrator

# ================================================== Experiment Configuration ==================================================

ALLOCATION_METHODS = ["gcbba", "sga", "cbba"]
COMM_RANGES = [3, 5, 8, 13, 20, 45]
TASKS_PER_INDUCT = [2, 5, 10]
SEEDS = [0, 1, 2, 3, 4]

# Simulation parameters
MAX_TIMESTEPS = 800
RERUN_INTERVAL = 10         # For dynamic replanning
STATIC_RERUN_INTERVAL = 9999  # Effectively disables replanning for static runs
SP_LIM = (1.0, 1.0)
STUCK_THRESHOLD = 15
PREDICTION_HORIZON = 5
MAX_PLAN_TIME = 400

# Whether to also run "static" (one-shot) variants
RUN_STATIC = True
RUN_DYNAMIC = True

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "gridworld_warehouse_small.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiments")

# ================================================== Experiment Runner ==================================================

def run_single_experiment(config_path, allocation_method, comm_range, tasks_per_induct, seed, rerun_interval, max_timesteps, label=""):
    """
    Run a single experiment with the given parameters and return the results as a dictionary.
    """
    
    np.random.seed(seed)  # Set seed for reproducibility
    
    try:
        # Initialize orchestrator with specified allocation method and parameters
        orchestrator = IntegrationOrchestrator(
            config_path=config_path,
            tasks_per_induct_station=tasks_per_induct,
            comm_range=comm_range,
            sp_lim=SP_LIM,
            rerun_interval=rerun_interval,
            stuck_threshold=STUCK_THRESHOLD,
            prediction_horizon=PREDICTION_HORIZON,
            max_plan_time=MAX_PLAN_TIME,
            allocation_method=allocation_method
        )

        total_tasks = len(orchestrator.all_task_ids)

        # Run the simulation
        t_start = time.time()
        orchestrator.run_simulation(timesteps=max_timesteps)
        t_end = time.time()

        completed = len(orchestrator.completed_task_ids)
        completion_rate = completed / total_tasks if total_tasks > 0 else 0
        wall_time = t_end - t_start
        final_timestep = orchestrator.current_timestep

        return {
            "allocation_method": allocation_method,
            "comm_range": comm_range,
            "tasks_per_induct": tasks_per_induct,
            "seed": seed,
            "rerun_interval": rerun_interval,
            "replanning": "dynamic" if rerun_interval < STATIC_RERUN_INTERVAL else "static",
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "completion_rate": completion_rate,
            "final_timestep": final_timestep,
            "wall_time_sec": wall_time,
            "label": label,
            "error": ""
        }

    except Exception as e:
        print(f"Error running experiment with allocation_method={allocation_method}, comm_range={comm_range}, tasks_per_induct={tasks_per_induct}, seed={seed}")
        traceback.print_exc()
        return {
            "allocation_method": allocation_method,
            "comm_range": comm_range,
            "tasks_per_induct": tasks_per_induct,
            "seed": seed,
            "rerun_interval": rerun_interval,
            "replanning": "dynamic" if rerun_interval < STATIC_RERUN_INTERVAL else "static",
            "total_tasks": -1,
            "completed_tasks": -1,
            "completion_rate": -1,
            "final_timestep": -1,
            "wall_time_sec": -1,
            "label": label,
            "error": str(e)
        }
    
def save_results_to_csv(results, output_path):
    """
    Save a list of result dictionaries to a CSV file.
    """
    if not results:
        print("No results to save.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    field_names = results[0].keys()

    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {output_path}")

# ================================================== Main ==================================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.csv")

    experiments = []

    for allocation_method, comm_range, tasks_per_induct, seed in product(ALLOCATION_METHODS, COMM_RANGES, TASKS_PER_INDUCT, SEEDS):
        if RUN_STATIC:
            experiments.append({
                "allocation_method": allocation_method,
                "comm_range": comm_range,
                "tasks_per_induct": tasks_per_induct,
                "seed": seed,
                "rerun_interval": STATIC_RERUN_INTERVAL,
                "label": f"{allocation_method}_static_cr{comm_range}_tpi{tasks_per_induct}_seed{seed}"
            })
        
        if RUN_DYNAMIC:
            experiments.append({
                "allocation_method": allocation_method,
                "comm_range": comm_range,
                "tasks_per_induct": tasks_per_induct,
                "seed": seed,
                "rerun_interval": RERUN_INTERVAL,
                "label": f"{allocation_method}_dynamic_cr{comm_range}_tpi{tasks_per_induct}_seed{seed}"
            })

    total = len(experiments)
    print(f"Running {total} experiments ({len(ALLOCATION_METHODS)} methods × "
        f"{len(COMM_RANGES)} ranges × {len(TASKS_PER_INDUCT)} loads × "
        f"{len(SEEDS)} seeds × {'static+dynamic' if RUN_STATIC and RUN_DYNAMIC else 'single'})")
    print(f"Results will be saved to: {output_path}\n")

    results = []
    for idx, exp in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{total}] {exp['label']}")
        print(f"{'='*60}")

        result = run_single_experiment(
            config_path=CONFIG_PATH,
            allocation_method=exp["allocation_method"],
            comm_range=exp["comm_range"],
            tasks_per_induct=exp["tasks_per_induct"],
            seed=exp["seed"],
            rerun_interval=exp["rerun_interval"],
            max_timesteps=MAX_TIMESTEPS,
            label=exp["label"]
        )
        results.append(result)

        # Print summary
        if result["error"]:
            print(f"  ❌ ERROR: {result['error']}")
        else:
            print(f"  ✓ {result['completed_tasks']}/{result['total_tasks']} tasks "
                f"({result['completion_rate']*100:.1f}%) in {result['final_timestep']} steps, "
                f"{result['wall_time_sec']:.1f}s wall time")
            
        # Save results after each experiment to avoid data loss
        save_results_to_csv(results, output_path)

    print(f"\n{'='*60}")
    print(f"All {total} experiments complete. Results: {output_path}")
    print(f"{'='*60}")