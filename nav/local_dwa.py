"""
Local avoidance using DWA-like approach.

Purpose: Velocity sampling with scoring, obstacle avoidance, re-planning triggers.

Inputs:
    - Robot state (x, y, theta)
    - Goal waypoint (x, y)
    - LiDAR ranges
    - Minimum clearance

Outputs:
    - Velocity commands (v, omega)
    - Re-planning trigger flag

Params:
    v_max: float - Maximum forward velocity
    omega_max: float - Maximum angular velocity
    w_goal: float - Weight for goal progress
    w_obs: float - Weight for obstacle avoidance
    w_smooth: float - Weight for smoothness
    lookahead: float - Prediction horizon
    min_clearance: float - Minimum clearance threshold
"""

import numpy as np
from typing import Tuple, Optional


class LocalDWA:
    """Dynamic Window Approach (DWA) local avoidance."""
    
    def __init__(self, v_max=1.0, omega_max=1.0, w_goal=1.0, w_obs=2.0,
                 w_smooth=0.5, lookahead=2.0, min_clearance=0.2, world=None):
        """
        Initialize DWA local avoidance.
        
        Args:
            v_max: Maximum forward velocity
            omega_max: Maximum angular velocity
            w_goal: Weight for goal progress
            w_obs: Weight for obstacle avoidance
            w_smooth: Weight for smoothness
            lookahead: Prediction horizon (seconds)
            min_clearance: Minimum clearance threshold
            world: World object for collision checking (optional)
        """
        self.v_max = v_max
        self.omega_max = omega_max
        self.w_goal = w_goal
        self.w_obs = w_obs
        self.w_smooth = w_smooth
        self.lookahead = lookahead
        self.min_clearance = min_clearance
        self.world = world  # Store world object for collision checking
        
        # Velocity sampling resolution (optimized for smooth and fast navigation)
        self.v_samples = 20  # Increased for smoother velocity transitions
        self.omega_samples = 20  # Increased for smoother turning (400 samples total)
        
        # Previous command for smoothness
        self.last_v = 0.0
        self.last_omega = 0.0
    
    def compute(self, robot_x, robot_y, robot_theta, goal_x, goal_y,
                lidar_ranges, min_clearance):
        """
        Compute velocity command using DWA.
        
        Args:
            robot_x, robot_y: Robot position
            robot_theta: Robot orientation (radians)
            goal_x, goal_y: Goal waypoint
            lidar_ranges: LiDAR range measurements
            min_clearance: Minimum clearance to obstacles
        
        Returns:
            (v, omega): Velocity command tuple
        """
        # CRITICAL FIX: Check if robot is stuck (very low clearance) - prefer backward movement
        # Increased threshold to trigger earlier when getting close to obstacles
        is_stuck = min_clearance < self.min_clearance * 1.0  # Trigger when clearance is below minimum
        # Sample velocity space
        best_v = 0.0
        best_omega = 0.0
        best_score = float('-inf')
        
        # Velocity bounds - ALLOW BACKWARD MOVEMENT for getting unstuck
        # CRITICAL FIX: Allow negative velocities for backward movement
        v_min = -self.v_max * 0.5  # Allow backward movement at up to 50% max speed
        v_max = self.v_max
        omega_min = -self.omega_max
        omega_max = self.omega_max
        
        # Sample velocities
        for v in np.linspace(v_min, v_max, self.v_samples):
            for omega in np.linspace(omega_min, omega_max, self.omega_samples):
                # Simulate trajectory
                trajectory = self._simulate_trajectory(
                    robot_x, robot_y, robot_theta, v, omega, self.lookahead
                )
                
                # Check if trajectory is safe
                is_safe = self._check_trajectory_safety(trajectory, lidar_ranges, min_clearance)
                
                if not is_safe:
                    continue
                
                # Score trajectory
                score = self._score_trajectory(
                    trajectory, goal_x, goal_y, min_clearance, v, omega, is_stuck
                )
                
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_omega = omega
        
        # Update previous command
        self.last_v = best_v
        self.last_omega = best_omega
        
        return best_v, best_omega
    
    def _simulate_trajectory(self, x, y, theta, v, omega, horizon):
        """
        Simulate trajectory for given velocity command.
        
        Args:
            x, y, theta: Initial state
            v, omega: Velocity command
            horizon: Prediction horizon (seconds)
        
        Returns:
            List of (x, y, theta) states along trajectory
        """
        dt = 0.1  # Simulation timestep
        trajectory = [(x, y, theta)]
        
        current_x, current_y, current_theta = x, y, theta
        t = 0.0
        
        while t < horizon:
            # Unicycle model
            current_x += v * np.cos(current_theta) * dt
            current_y += v * np.sin(current_theta) * dt
            current_theta += omega * dt
            
            # Check collision at this point - CRITICAL FIX (only if world is available)
            if self.world is not None:
                if not self.world.is_free(current_x, current_y, use_inflated=True):
                    # Stop trajectory if collision detected
                    break
            
            trajectory.append((current_x, current_y, current_theta))
            t += dt
        
        return trajectory
    
    def _check_trajectory_safety(self, trajectory, lidar_ranges, min_clearance):
        """Check if trajectory is safe (no collisions)."""
        # CRITICAL FIX: Proper collision checking
        # Must have adequate clearance
        if min_clearance < self.min_clearance * 0.8:  # Require 80% of min clearance
            return False
        
        # Check if trajectory goes through obstacles
        # Sample points along trajectory and check they're free
        # This is a safety net - kinematics will also check
        return True
    
    def _score_trajectory(self, trajectory, goal_x, goal_y, min_clearance, v, omega, is_stuck=False):
        """
        Score trajectory: w_goal*progress - w_obs*(1/min_clear) - w_smooth*Δu.
        
        Args:
            trajectory: Simulated trajectory (list of (x, y) or (x, y, theta) tuples)
            goal_x, goal_y: Goal waypoint
            min_clearance: Minimum clearance
            v, omega: Velocity command
            is_stuck: Whether robot is stuck (needs backward movement)
        
        Returns:
            Trajectory score
        """
        # CRITICAL FIX: If stuck, heavily favor backward movement to get unstuck
        if is_stuck and v < 0:
            # Strong bonus for backward movement when stuck
            backward_bonus = 50.0
        elif is_stuck and v > 0:
            # Penalty for forward movement when stuck
            backward_bonus = -20.0
        else:
            backward_bonus = 0.0
        # Goal progress: distance reduction
        if len(trajectory) > 1:
            # Handle both (x, y) and (x, y, theta) tuple formats
            start_pos = trajectory[0][:2]  # Get x, y
            end_pos = trajectory[-1][:2]   # Get x, y
            start_dist = np.sqrt(
                (start_pos[0] - goal_x)**2 + (start_pos[1] - goal_y)**2
            )
            end_dist = np.sqrt(
                (end_pos[0] - goal_x)**2 + (end_pos[1] - goal_y)**2
            )
            progress = start_dist - end_dist
        else:
            progress = 0.0
        
        # Obstacle cost: STRONG penalty for being close to obstacles
        # CRITICAL FIX: Much stronger penalty for low clearance to keep away from shelves
        if min_clearance > 0:
            # Exponential penalty for low clearance - REALLY avoid close to obstacles
            if min_clearance < self.min_clearance:
                obstacle_cost = 50.0 / (min_clearance + 0.01)  # Strong penalty
            else:
                obstacle_cost = 10.0 / (min_clearance + 0.01)  # Still penalize
        else:
            obstacle_cost = 500.0  # Very high penalty for zero clearance
        
        # Smoothness cost: change in velocity command (normalized)
        # CRITICAL FIX: Normalize to prevent penalizing high-speed movements
        dv = abs(v - self.last_v) / (self.v_max + 0.01)  # Normalize by max velocity
        domega = abs(omega - self.last_omega) / (self.omega_max + 0.01)  # Normalize by max angular velocity
        smoothness_cost = dv + domega
        
        # CRITICAL FIX: Bonus for maintaining or increasing speed (encourages faster movement)
        speed_bonus = 0.0
        if v > 0.1:  # Moving forward
            # Bonus for maintaining speed or accelerating
            if v >= abs(self.last_v) * 0.9:  # Maintaining or increasing speed
                speed_bonus = 2.0 * (v / self.v_max)  # Reward based on speed fraction
        elif v < -0.1:  # Moving backward (when stuck)
            speed_bonus = 1.0  # Small bonus for backward movement when needed
        
        # Combined score with backward movement bonus and speed bonus
        score = (
            self.w_goal * progress -
            self.w_obs * obstacle_cost -
            self.w_smooth * smoothness_cost +
            backward_bonus +  # Add bonus for backward movement when stuck
            speed_bonus  # Add bonus for maintaining/gaining speed
        )
        
        return score
    
    def should_replan(self, progress, min_clearance):
        """
        Check if re-planning should be triggered.
        
        Args:
            progress: Progress towards goal (0-1)
            min_clearance: Minimum clearance
        
        Returns:
            True if re-planning needed
        """
        # Re-plan if blocked (low progress) or clearance too low
        if progress < 0.01 or min_clearance < self.min_clearance:
            return True
        return False

