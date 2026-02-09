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

class AgentState:
    """
    Class to manage executin state of single agent

    GCBBA_Agent: Handles bidding, consensus, and task assignment logic
    AgentState: Handles actual execution state of assigned tasks
     - Tracks task lifecycle: planned -> executing -> completed
     - Stores path and timing info for each task
     - Provides methods to update execution state and retrieve current task info
    """

    def __init__(self, agent_id: int, initial_position: Tuple[float, float, float], speed: float):
        self.agent_id = agent_id
        self.pos = np.array(initial_position, dtype=np.float32)
        self.speed = speed

        # Task Lifecycle Management
        self.planned_tasks: List[TaskExecutionInfo] = []
        self.current_task: Optional[TaskExecutionInfo] = None
        self.completed_tasks: List[TaskExecutionInfo] = []

        # Path Tracking
        self.current_path: Optional[List[Tuple[int, int, int]]] = None
        self.current_path_index: int = 0

        # state tracking
        self.is_idle = True
        self.is_stuck = False # flag to indicate if agent is stuck (e.g. due to collision or path blockage)

        self.position_history: List[Tuple[float, float, float]] = [initial_position] # track position history 

        self.current_timestep: int = 0

    def update_from_gcbba(self, assigned_tasks: List[Dict], current_timestep: int):
        """
        Update agent state based on GCBBA task assignments
        - assigned_tasks: List of task dicts assigned to this agent by GCBBA
        - current_timestep: Current simulation timestamp for timing info
        """
        self.current_timestep = current_timestep

        new_planned_tasks = []
        for task in assigned_tasks:
            task_info = TaskExecutionInfo(
                task_id=task['task_id'],
                induct_pos=tuple(task['induct_pos']),
                eject_pos=tuple(task['eject_pos']),
                state=TaskState.PLANNED,
                assigned_time=current_timestep
            )
            new_planned_tasks.append(task_info)

        self.planned_tasks = new_planned_tasks

        if self.is_idle and len(self.planned_tasks) > 0:
            self.is_idle = False
    
    def assign_path(self, path: List[Tuple[int, int, int]]):
        """
        Set the current path for the agent to execute the current task
        """

        if self.current_task is None and len(self.planned_tasks) > 0:
            # Start executing the first planned task
            self.current_task = self.planned_tasks.pop(0)
            self.current_task.state = TaskState.EXECUTING

        if self.current_task is not None:
            # Update the current task with the new path and reset path index
            self.current_task.path = path
            self.current_task.current_path_index = 0
            
            # Update the agent's current path and reset path index for execution
            self.current_path = path
            self.current_path_index = 0

    def step(self, timestep:int) -> bool:
        """
        Execute one step of the agent's current task based on the assigned path
        """

        if self.current_path is None or len(self.current_path) == 0:
            self.position_history.append((self.pos[0], self.pos[1], self.pos[2], timestep))
            return False
        
        # If the agent is currently executing a task, move along the assigned path
        if self.current_path_index < len(self.current_path):
            next_pos = self.current_path[self.current_path_index]
            self.pos = np.array(next_pos, dtype=np.float32)
            self.current_path_index += 1

            self.position_history.append((self.pos[0], self.pos[1], self.pos[2], timestep))

            if self.current_path_index >= len(self.current_path):
                return self._complete_current_task(timestep)
        
        return False

    def _complete_current_task(self, timestep: int) -> bool:
        """
        Mark the current task as completed and update state
        Then transition to the next planned task if available
        """

        if self.current_task is not None:
            self.current_task.state = TaskState.COMPLETED
            self.current_task.completion_time = timestep
            self.completed_tasks.append(self.current_task)

            # Reset current task and path info
            self.current_task = None
            self.current_path = None
            self.current_path_index = 0

            # Check if there are more planned tasks to execute
            if len(self.planned_tasks) > 0:
                self.is_idle = False
            else:
                self.is_idle = True

            return True
        
        return False
    
    def get_predicted_position(self, steps_ahead: int = 5) -> Tuple[float, float, float]:
        """
        Predict the agent's future position based on current path and speed
        - steps_ahead: Number of steps to predict into the future
        """ 
        if self.current_path is None or len(self.current_path) == 0:
            return (self.pos[0], self.pos[1], self.pos[2])
        
        predicted_idx = min(self.current_path_index + steps_ahead, len(self.current_path) - 1)

        if predicted_idx < len(self.current_path):
            predicted_pos = self.current_path[predicted_idx]
            return predicted_pos
        else:
            return self.current_path[-1]