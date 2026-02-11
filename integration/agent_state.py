"""
AgentState: 

Class to manage the execution state of an agent seperately from GCBBA logic
task lifecycle: planned -> executing -> completed
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Iterable
from enum import Enum
from dataclasses import dataclass

class TaskState(Enum):
    PLANNED = "planned"
    EXECUTING = "executing"
    COMPLETED = "completed"

@dataclass
class TaskExecutionInfo:
    task_id: int
    induct_pos: Tuple[int, int, int]
    eject_pos: Tuple[int, int, int]
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

    def __init__(self, agent_id: int, initial_position: Tuple[int, int, int], speed: float = 1.0):
        self.agent_id = agent_id
        self.pos = np.array(initial_position, dtype=np.int32)
        self.speed = speed

        # Task Lifecycle Management
        self.planned_tasks: List[TaskExecutionInfo] = []
        self.current_task: Optional[TaskExecutionInfo] = None
        self.completed_tasks: List[TaskExecutionInfo] = []

        # Path Tracking
        self.current_path: Optional[List[Tuple[int, int, int]]] = None
        self.current_path_index: int = 0
        self.task_phase: str = "to_induct" # or "to_eject" to track which part of the task is being executed

        # state tracking
        self.is_idle = True
        self.is_stuck = False # flag to indicate if agent is stuck (e.g. due to collision or path blockage)
        self.needs_new_path = False # Used by orchestrator to know when to call path planner for this agent

        self.position_history: List[Tuple[int, int, int, int]] = [
            (initial_position[0], initial_position[1], initial_position[2], 0)
        ]

        self.current_timestep: int = 0

    def update_from_gcbba(self, assigned_tasks: List[Dict], current_timestep: int):
        """
        Update agent state based on GCBBA task assignments
        - assigned_tasks: List of task dicts assigned to this agent by GCBBA
        - current_timestep: Current simulation timestamp for timing info
        """
        self.current_timestep = current_timestep

        executing_task_id = self.current_task.task_id if self.current_task else None

        new_planned_tasks = []
        for task in assigned_tasks:
            if task['task_id'] == executing_task_id:
                # Skip already executing task
                continue
            
            task_info = TaskExecutionInfo(
                task_id=task['task_id'],
                induct_pos=self._to_grid_pos(task['induct_pos']),
                eject_pos=self._to_grid_pos(task['eject_pos']),
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

        If no current task is executing, promotes the first planned task to executing.
        Sets start_time on the task when it first begins executing.
        Clears the needs_new_path flag since a path has been provided.
        """

        if self.current_task is None and len(self.planned_tasks) > 0:
            # Start executing the first planned task
            self.current_task = self.planned_tasks.pop(0)
            self.current_task.state = TaskState.EXECUTING
            self.current_task.start_time = self.current_timestep
            self.task_phase = "to_induct" 

        if self.current_task is not None:
            # Update the current task with the new path and reset path index
            self.current_task.path = path
            self.current_task.current_path_index = 0
            
            # Update the agent's current path and reset path index for execution
            self.current_path = path
            self.current_path_index = 0
            self.needs_new_path = False

    def step(self, timestep:int) -> bool:
        """
        Execute one step of the agent's current task based on the assigned path

        Handles two-phase task execution:
            - to_induct phase: agent follows path to induct station.
              On arrival, flips to to_eject and sets needs_new_path = True.
            - to_eject phase: agent follows path to eject station.
              On arrival, completes the task.
        """

        self.current_timestep = timestep

        if self.current_path is None or len(self.current_path) == 0:
            self.position_history.append((self.pos[0], self.pos[1], self.pos[2], timestep))
            return False
        
        # If the agent is currently executing a task, move along the assigned path
        if self.current_path_index < len(self.current_path):
            next_pos = self.current_path[self.current_path_index]
            self.pos = np.array(next_pos, dtype=np.int32)
            self.current_path_index += 1

            if self.current_task is not None:
                self.current_task.current_path_index = self.current_path_index
                
            self.position_history.append((self.pos[0], self.pos[1], self.pos[2], timestep))

            # Reached end of the current path
            if self.current_path_index >= len(self.current_path):
                if self.task_phase == "to_induct":
                    # Arrived at induct station, now need path to eject station
                    self.task_phase = "to_eject"
                    self.current_path = None
                    self.current_path_index = 0
                    self.needs_new_path = True
                    return False
                elif self.task_phase == "to_eject":
                    # Arrived at eject station, task is complete
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
            self.task_phase = "to_induct"
            self.needs_new_path = False

            # Check if there are more planned tasks to execute
            if len(self.planned_tasks) > 0:
                self.is_idle = False
                self.needs_new_path = True 
            else:
                self.is_idle = True

            return True
        
        return False
    
    def get_position(self) -> Tuple[int, int, int]:
        """
        Get the current position of the agent
        """
        if self.current_path is not None and self.current_path_index > 0:
            # Return the last position on the path that the agent has reached
            return self.current_path[self.current_path_index - 1]
        return (int(self.pos[0]), int(self.pos[1]), int(self.pos[2]))
    
    def get_predicted_position(self, steps_ahead: int = 5) -> Tuple[int, int, int]:
        """
        Predict the agent's future position based on current path and speed
        - steps_ahead: Number of steps to predict into the future
        """ 
        if self.current_path is None or len(self.current_path) == 0:
            return (int(self.pos[0]), int(self.pos[1]), int(self.pos[2]))
        
        predicted_idx = min(self.current_path_index + steps_ahead, len(self.current_path) - 1)

        return self.current_path[predicted_idx]
        
    def get_current_goal(self) -> Optional[Tuple[int, int, int]]:
        """
        Get the current goal position of the executing task

        - If executing to_induct: returns induct position
        - If executing to_eject: returns eject position
        - If idle with planned tasks: returns first planned task's induct position
        - If idle with no tasks: returns None
        """
        if self.current_task is not None:
            if self.task_phase == "to_induct":
                return self.current_task.induct_pos
            elif self.task_phase == "to_eject":
                return self.current_task.eject_pos

        elif len(self.planned_tasks) > 0:
            return self.planned_tasks[0].induct_pos
        
        else:
            return None
        
    def get_next_task_goal(self) -> Optional[Tuple[int, int, int]]:
        """
        Get the next task's goal position after the current task
        """
        if len(self.planned_tasks) > 0:
            return self.planned_tasks[0].induct_pos
        else:
            return None
    
    def detect_stuck(self, stuck_threshold: int = 5) -> bool:
        """
        Detect if the agent is stuck based on lack of position change over time
        - stuck_threshold: Number of timesteps with no movement to consider as stuck
        """
        if self.is_idle:
            self.is_stuck = False
            return False
        
        if len(self.position_history) < stuck_threshold:
            self.is_stuck = False
            return False
        
        recent_positions = self.position_history[-stuck_threshold:]
        first_pos = recent_positions[0][:3]
        
        for pos in recent_positions[1:]:
            if pos[:3] != first_pos:
                return False
        
        self.is_stuck = True
        return True
    
    def has_tasks(self) -> bool:
        """
        Check if the agent has any tasks (planned or executing)
        """
        return self.current_task is not None or len(self.planned_tasks) > 0
    
    def get_status_summary(self) -> Dict:
        """
        Get summary of agent's current state for debugging/logging.
        """
        return {
            'agent_id': self.agent_id,
            'position': (int(self.pos[0]), int(self.pos[1]), int(self.pos[2])),
            'is_idle': self.is_idle,
            'is_stuck': self.is_stuck,
            'task_phase': self.task_phase,
            'needs_new_path': self.needs_new_path,
            'current_task': self.current_task.task_id if self.current_task else None,
            'num_planned': len(self.planned_tasks),
            'num_completed': len(self.completed_tasks),
            'path_progress': f"{self.current_path_index}/{len(self.current_path) if self.current_path else 0}",
            'timestep': self.current_timestep
        }
    
    def __repr__(self):
        status = "IDLE" if self.is_idle else f"BUSY({self.task_phase})"
        task_info = f"task={self.current_task.task_id}" if self.current_task else "no task"
        return f"Agent{self.agent_id}@{tuple(self.pos)} [{status}] {task_info}"

    def _to_grid_pos(self, pos: Iterable) -> Tuple[int, int, int]:
        pos_list = list(pos)
        if len(pos_list) != 3:
            raise ValueError(f"Expected 3D position, got {len(pos_list)} values")
        return (int(pos_list[0]), int(pos_list[1]), int(pos_list[2]))