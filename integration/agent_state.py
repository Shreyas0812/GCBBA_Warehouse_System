"""
AgentState: 

Class to manage the execution state of an agent seperately from GCBBA logic
task lifecycle: planned -> executing -> completed
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum
from dataclasses import dataclass

class TaskState(Enum):
    PLANNED = "planned"
    EXECUTING = "executing"
    COMPLETED = "completed"

@dataclass
class TaskExecutionInfo:
    task_id: int
    induct_pos: Tuple[float, float]
    eject_pos: Tuple[float, float]
    state: TaskState
    path: Optional[List[Tuple[int, int, int]]] = None
    current_path_index: int = 0
    assigned_time: Optional[float] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None