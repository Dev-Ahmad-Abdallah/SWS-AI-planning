"""
Task FSM (Finite State Machine) for mission execution.

Purpose: Task executive with realistic warehouse missions: 
    PICKUP → GOTO_BAY → LOAD → GOTO_DELIVERY → UNLOAD → GOTO_DOCK → DONE.

Inputs:
    - Robot state (x, y, theta)
    - Pickup locations (bays/aisles)
    - Delivery locations (bays/aisles)
    - Dock position

Outputs:
    - Current goal waypoint
    - Mission completion status
    - Current state

Params:
    bays: List of (x, y) pickup/delivery positions
    dock: (x, y) dock position
"""

from typing import List, Tuple, Optional
from enum import Enum


class TaskState(Enum):
    """Task FSM states."""
    IDLE = "IDLE"
    GOTO_PICKUP = "GOTO_PICKUP"  # Going to pickup location
    LOAD = "LOAD"  # Loading item
    GOTO_DELIVERY = "GOTO_DELIVERY"  # Going to delivery location
    UNLOAD = "UNLOAD"  # Unloading item
    GOTO_BAY = "GOTO_BAY"  # Legacy: going to bay
    INSPECT = "INSPECT"  # Legacy: inspecting
    GOTO_DOCK = "GOTO_DOCK"
    DONE = "DONE"
    WAIT = "WAIT"
    TURN_IN_PLACE = "TURN_IN_PLACE"
    ALT_AISLE = "ALT_AISLE"


class TaskFSM:
    """Task Finite State Machine with realistic warehouse missions."""
    
    def __init__(self, bays: List[Tuple[float, float]], dock: Tuple[float, float], mission_type: str = "delivery"):
        """
        Initialize task FSM.
        
        Args:
            bays: List of (x, y) pickup/delivery positions
            dock: (x, y) dock position
            mission_type: "inspect" (old) or "delivery" (new realistic missions)
        """
        self.bays = bays
        self.dock = dock
        self.mission_type = mission_type
        
        # State machine
        self.state = TaskState.IDLE
        self.current_task_idx = 0  # Current pickup-delivery pair
        self.current_goal = None
        self.has_load = False  # Whether robot is carrying item
        
        # Create realistic missions: list of (pickup, delivery) pairs
        # Each mission: pick up from one aisle, deliver to another
        self.missions = []
        if mission_type == "delivery" and len(bays) >= 2:
            # Create pickup-delivery pairs
            for i in range(0, len(bays) - 1, 2):
                if i + 1 < len(bays):
                    self.missions.append((bays[i], bays[i + 1]))  # Pickup from bay[i], deliver to bay[i+1]
            # If odd number of bays, add dock as final delivery
            if len(bays) % 2 == 1:
                self.missions.append((bays[-1], dock))
        else:
            # Legacy inspect missions
            self.missions = [(bay, None) for bay in bays]  # Just inspect each bay
        
        # Timeouts
        self.load_timeout = 2.0  # seconds to load
        self.unload_timeout = 2.0  # seconds to unload
        self.load_start_time = 0.0
        self.unload_start_time = 0.0
        self.inspect_timeout = 3.0  # Legacy: seconds to inspect
        self.inspect_start_time = 0.0
        self.wait_timeout = 5.0
        self.wait_start_time = 0.0
        
        # Deadlock detection
        self.last_position = None
        self.stuck_time = 0.0
        self.stuck_threshold = 10.0  # seconds
        self.stuck_distance = 0.1  # meters
        
        # Mission status
        self.mission_complete = False
        
    def update(self, robot_x, robot_y, robot_theta, sim_time=0.0):
        """
        Update task FSM - can be called less frequently for performance.
        
        Args:
            robot_x, robot_y: Robot position
            robot_theta: Robot orientation
            sim_time: Current simulation time
        """
        # Deadlock detection
        if self.last_position is not None:
            dist = (
                (robot_x - self.last_position[0])**2 +
                (robot_y - self.last_position[1])**2
            )**0.5
            
            if dist < self.stuck_distance:
                self.stuck_time += 0.01667  # Assume ~60 FPS dt
            else:
                self.stuck_time = 0.0
        else:
            self.stuck_time = 0.0
        
        self.last_position = (robot_x, robot_y)
        self._last_update_time = sim_time
        
        # State machine logic - REALISTIC WAREHOUSE MISSIONS
        if self.mission_type == "delivery" and self.missions:
            # NEW: Realistic pickup-delivery missions
            if self.state == TaskState.IDLE:
                if self.current_task_idx < len(self.missions):
                    pickup, _ = self.missions[self.current_task_idx]
                    self.state = TaskState.GOTO_PICKUP
                    self.current_goal = pickup
                    print(f"Mission {self.current_task_idx + 1}: Going to pickup location")
            
            elif self.state == TaskState.GOTO_PICKUP:
                if self.current_goal:
                    dist = ((robot_x - self.current_goal[0])**2 + (robot_y - self.current_goal[1])**2)**0.5
                    if dist < 0.5:  # Reached pickup
                        self.state = TaskState.LOAD
                        self.load_start_time = sim_time
                        print(f"Mission {self.current_task_idx + 1}: Loading item...")
            
            elif self.state == TaskState.LOAD:
                if sim_time - self.load_start_time >= self.load_timeout:
                    self.has_load = True
                    pickup, delivery = self.missions[self.current_task_idx]
                    self.state = TaskState.GOTO_DELIVERY
                    self.current_goal = delivery
                    print(f"Mission {self.current_task_idx + 1}: Item loaded, going to delivery location")
            
            elif self.state == TaskState.GOTO_DELIVERY:
                if self.current_goal:
                    dist = ((robot_x - self.current_goal[0])**2 + (robot_y - self.current_goal[1])**2)**0.5
                    if dist < 0.5:  # Reached delivery
                        self.state = TaskState.UNLOAD
                        self.unload_start_time = sim_time
                        print(f"Mission {self.current_task_idx + 1}: Unloading item...")
            
            elif self.state == TaskState.UNLOAD:
                if sim_time - self.unload_start_time >= self.unload_timeout:
                    self.has_load = False
                    self.current_task_idx += 1
                    
                    if self.current_task_idx >= len(self.missions):
                        # All missions complete, go to dock
                        self.state = TaskState.GOTO_DOCK
                        self.current_goal = self.dock
                        print("All missions complete, returning to dock")
                    else:
                        # Start next mission
                        pickup, _ = self.missions[self.current_task_idx]
                        self.state = TaskState.GOTO_PICKUP
                        self.current_goal = pickup
                        print(f"Mission {self.current_task_idx + 1}: Starting next delivery task")
        else:
            # LEGACY: Simple inspect missions
            if self.state == TaskState.IDLE:
                if self.current_task_idx < len(self.bays):
                    self.state = TaskState.GOTO_BAY
                    self.current_goal = self.bays[self.current_task_idx]
                    print(f"Task: Going to bay {self.current_task_idx + 1}")
            
            elif self.state == TaskState.GOTO_BAY:
                if self.current_goal:
                    dist = ((robot_x - self.current_goal[0])**2 + (robot_y - self.current_goal[1])**2)**0.5
                    if dist < 0.5:  # Reached bay
                        self.state = TaskState.INSPECT
                        self.inspect_start_time = sim_time
                        print(f"Task: Inspecting bay {self.current_task_idx + 1}")
            
            elif self.state == TaskState.INSPECT:
                if sim_time - self.inspect_start_time >= self.inspect_timeout:
                    self.current_task_idx += 1
                    if self.current_task_idx >= len(self.bays):
                        self.state = TaskState.GOTO_DOCK
                        self.current_goal = self.dock
                        print("Task: All bays visited, going to dock")
                    else:
                        self.state = TaskState.GOTO_BAY
                        self.current_goal = self.bays[self.current_task_idx]
                        print(f"Task: Going to bay {self.current_task_idx + 1}")
        
        # Common states (for both mission types)
        if self.state == TaskState.GOTO_DOCK:
            # Check if dock reached
            if self.current_goal:
                dist = (
                    (robot_x - self.current_goal[0])**2 +
                    (robot_y - self.current_goal[1])**2
                )**0.5
                
                if dist < 0.5:  # Reached dock
                    self.state = TaskState.DONE
                    self.mission_complete = True
                    self.current_goal = None
                    print("Task: Mission complete!")
        
        if self.state == TaskState.WAIT:
            # Wait for obstacle to clear
            if sim_time - self.wait_start_time >= self.wait_timeout:
                # Resume previous state
                if self.current_task_idx < len(self.bays):
                    if self.mission_type == "delivery":
                        pickup, _ = self.missions[self.current_task_idx] if self.current_task_idx < len(self.missions) else (self.bays[0], None)
                        self.state = TaskState.GOTO_PICKUP
                        self.current_goal = pickup
                    else:
                        self.state = TaskState.GOTO_BAY
                        self.current_goal = self.bays[self.current_task_idx]
                else:
                    self.state = TaskState.GOTO_DOCK
                    self.current_goal = self.dock
        
        if self.state == TaskState.TURN_IN_PLACE:
            # Turn in place (could implement actual turning logic)
            if self.mission_type == "delivery" and self.current_task_idx < len(self.missions):
                pickup, _ = self.missions[self.current_task_idx]
                self.state = TaskState.GOTO_PICKUP
                self.current_goal = pickup
            else:
                self.state = TaskState.GOTO_BAY
        
        if self.state == TaskState.ALT_AISLE:
            # Select alternate aisle
            if self.mission_type == "delivery" and self.current_task_idx < len(self.missions):
                pickup, _ = self.missions[self.current_task_idx]
                self.state = TaskState.GOTO_PICKUP
                self.current_goal = pickup
            elif self.current_task_idx < len(self.bays):
                self.state = TaskState.GOTO_BAY
                self.current_goal = self.bays[self.current_task_idx]
        
        # Check for stuck condition
        if self.stuck_time > self.stuck_threshold:
            print("Task: Deadlock detected, attempting recovery")
            self._recover_from_deadlock()
    
    def _recover_from_deadlock(self):
        """Recover from deadlock."""
        # Try alternate aisle or wait
        if self.state in [TaskState.GOTO_BAY, TaskState.GOTO_PICKUP, TaskState.GOTO_DELIVERY]:
            self.state = TaskState.WAIT
            self.wait_start_time = 0.0
        elif self.state == TaskState.GOTO_DOCK:
            self.state = TaskState.TURN_IN_PLACE
        
        self.stuck_time = 0.0
    
    def get_current_goal(self):
        """Get current goal waypoint."""
        return self.current_goal
    
    def is_done(self):
        """Check if mission is complete."""
        return self.mission_complete
    
    def reset(self):
        """Reset task FSM."""
        self.state = TaskState.IDLE
        self.current_task_idx = 0
        self.current_goal = None
        self.has_load = False
        self.mission_complete = False
        self.last_position = None
        self.stuck_time = 0.0
    
    @property
    def current_bay_idx(self):
        """Legacy compatibility: return current_task_idx."""
        return self.current_task_idx

