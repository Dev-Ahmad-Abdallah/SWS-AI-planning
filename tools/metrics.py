"""
Metrics tracker for KPIs.

Purpose: Track and compute KPIs during simulation run.

Inputs:
    - Path length
    - Min clearance
    - Planning time
    - Re-planning events
    - Shortest-path baseline

Outputs:
    - Finalized metrics dictionary with all KPIs

Params:
    shortest_path_baseline: float - Shortest path length for efficiency metric
"""

from typing import Dict, Optional


class MetricsTracker:
    """Metrics tracker for KPIs."""
    
    def __init__(self, shortest_path_baseline: Optional[float] = None):
        """
        Initialize metrics tracker.
        
        Args:
            shortest_path_baseline: Shortest path length for efficiency metric
        """
        self.shortest_path_baseline = shortest_path_baseline
        
        # Tracked metrics
        self.path_length = 0.0
        self.min_clearance = float('inf')
        self.replans = 0
        self.total_plan_time = 0.0
        self.plan_calls = 0
        self.has_collision_flag = False
    
    def update_path_length(self, path_len: float):
        """Update path length."""
        self.path_length = max(self.path_length, path_len)
    
    def update_min_clearance(self, clearance: float):
        """Update minimum clearance."""
        self.min_clearance = min(self.min_clearance, clearance)
    
    def record_replan(self):
        """Record a re-planning event."""
        self.replans += 1
    
    def record_plan_time(self, plan_time_ms: float):
        """Record planning time."""
        self.total_plan_time += plan_time_ms
        self.plan_calls += 1
    
    def record_collision(self):
        """Record a collision."""
        self.has_collision_flag = True
    
    def has_collision(self) -> bool:
        """Check if collision occurred."""
        return self.has_collision_flag
    
    def get_current_metrics(self) -> Dict:
        """Get current metrics."""
        return {
            "path_length": self.path_length,
            "min_clearance": self.min_clearance,
            "replans": self.replans,
            "plan_time_ms": self.total_plan_time / max(1, self.plan_calls),
            "has_collision": self.has_collision_flag
        }
    
    def finalize(self, sim_time: float, success: bool) -> Dict:
        """
        Finalize metrics and compute efficiency.
        
        Args:
            sim_time: Simulation time to goal
            success: Mission success (1 if completed without collision, else 0)
        
        Returns:
            Dictionary with all KPIs
        """
        # Compute efficiency
        if self.shortest_path_baseline is not None and self.shortest_path_baseline > 0:
            efficiency = self.path_length / self.shortest_path_baseline
        else:
            efficiency = 0.0
        
        # Average planning time
        avg_plan_time = self.total_plan_time / max(1, self.plan_calls)
        
        return {
            "t_goal": sim_time,
            "path_len": self.path_length,
            "min_clear": self.min_clearance if self.min_clearance < float('inf') else 0.0,
            "replans": self.replans,
            "cpu_ms": avg_plan_time,
            "success": 1 if success and not self.has_collision_flag else 0,
            "efficiency": efficiency
        }
    
    def reset(self):
        """Reset metrics tracker."""
        self.path_length = 0.0
        self.min_clearance = float('inf')
        self.replans = 0
        self.total_plan_time = 0.0
        self.plan_calls = 0
        self.has_collision_flag = False

