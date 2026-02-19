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