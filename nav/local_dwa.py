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
        
        # FIXED: Better balanced sampling for path following
        self.v_samples = 20  # Balanced: Enough for exploration, not too random
        self.omega_samples = 20  # Balanced: Good coverage (400 total)
        
        # Previous command for smoothness
        self.last_v = 0.0
        self.last_omega = 0.0
        
        # Stuck detection: Track consecutive low-velocity commands
        self.stuck_counter = 0
        self.stuck_threshold = 5  # REDUCED: Trigger stuck detection faster (reduced from 10)
        self.emergency_stuck_clearance = 0.3  # Emergency threshold - immediate backward movement
        
        # FIXED: Light smoothing - don't over-smooth and prevent path following
        self.alpha = 0.3  # REDUCED: Lighter smoothing for better responsiveness (was 0.7)
    
    def compute(self, robot_x, robot_y, robot_theta, goal_x, goal_y,
                lidar_ranges, min_clearance, actor_polygons=None):
        """
        Compute velocity command using DWA.
        
        Args:
            robot_x, robot_y: Robot position
            robot_theta: Robot orientation (radians)
            goal_x, goal_y: Goal waypoint
            lidar_ranges: LiDAR range measurements
            min_clearance: Minimum clearance to obstacles
            actor_polygons: List of moving actor polygons for dynamic obstacle avoidance (optional)
        
        Returns:
            (v, omega): Velocity command tuple
        """
        # IMPROVED: Better stuck detection - check clearance AND recent movement
        # CRITICAL: Detect stuck earlier and more aggressively
        clearance_stuck = min_clearance < self.min_clearance * 1.5  # Trigger much earlier (was 1.2)
        emergency_stuck = min_clearance < self.emergency_stuck_clearance  # Emergency threshold - immediate action
        
        # Also check if we've been stuck for multiple frames
        movement_threshold = 0.1  # INCREASED: Consider stuck if moving less than this (more sensitive)
        if abs(self.last_v) < movement_threshold and abs(self.last_omega) < movement_threshold:
            self.stuck_counter += 2  # Increment faster when not moving
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        # Stuck if: clearance too low OR stuck for multiple frames OR emergency clearance
        is_stuck = clearance_stuck or (self.stuck_counter > self.stuck_threshold) or emergency_stuck
        is_emergency = emergency_stuck  # Emergency flag for immediate backward movement
        
        # Sample velocity space with adaptive sampling (more samples near current velocity)
        best_v = 0.0
        best_omega = 0.0
        best_score = float('-inf')
        
        # Velocity bounds - ALLOW BACKWARD MOVEMENT for getting unstuck
        # IMPROVED: Allow much more backward movement when stuck or in emergency
        if is_emergency:
            v_min = -self.v_max * 0.85  # EMERGENCY: Fast backward movement (85% max speed)
        elif is_stuck:
            v_min = -self.v_max * 0.75  # More backward movement when stuck (75% max speed, increased from 60%)
        else:
            v_min = -self.v_max * 0.4  # Increased backward capability even when not stuck (was 0.3)
        v_max = self.v_max
        omega_min = -self.omega_max
        omega_max = self.omega_max
        
        # FIXED: Balanced sampling - favor current velocity but allow exploration for path following
        if is_stuck:
            # When stuck, sample uniformly to find escape routes
            v_samples = np.linspace(v_min, v_max, self.v_samples)
            omega_samples = np.linspace(omega_min, omega_max, self.omega_samples)
        else:
            # FIXED: Moderate bias toward current - allow exploration for path following
            # Compute desired heading toward goal for better path following
            dx = goal_x - robot_x
            dy = goal_y - robot_y
            goal_dist = np.sqrt(dx*dx + dy*dy)
            desired_theta = np.arctan2(dy, dx) if goal_dist > 0.1 else robot_theta
            
            # Angle difference to goal
            angle_to_goal = desired_theta - robot_theta
            # Normalize to [-pi, pi]
            while angle_to_goal > np.pi:
                angle_to_goal -= 2 * np.pi
            while angle_to_goal < -np.pi:
                angle_to_goal += 2 * np.pi
            
            # Sample velocities - slight bias toward current for smoothness
            v_center = np.clip(self.last_v, v_min, v_max)
            v_samples = np.linspace(v_min, v_max, self.v_samples)  # Uniform sampling for better path following
            
            # Sample angular velocities - bias toward goal direction when far from goal
            if abs(angle_to_goal) > 0.3:  # Far from goal heading - favor turning toward goal
                # Bias toward turning toward goal
                omega_preferred = np.clip(angle_to_goal * 2.0, omega_min, omega_max)
                # Create samples with bias toward preferred omega
                omega_samples = np.concatenate([
                    np.linspace(omega_min, omega_preferred - 0.5, self.omega_samples // 3),
                    np.linspace(omega_preferred - 0.5, omega_preferred + 0.5, self.omega_samples // 2),
                    np.linspace(omega_preferred + 0.5, omega_max, self.omega_samples - self.omega_samples // 3 - self.omega_samples // 2)
                ])
                omega_samples = np.unique(np.clip(omega_samples, omega_min, omega_max))
            else:
                # Close to goal heading - slight bias toward current omega
                omega_center = np.clip(self.last_omega, omega_min, omega_max)
                omega_range = omega_max - omega_min
                n_near = self.omega_samples // 3  # 1/3 near current
                n_far = self.omega_samples - n_near
                omega_samples = np.concatenate([
                    np.linspace(omega_min, max(omega_min, omega_center - 0.3 * omega_range), n_far // 2),
                    np.linspace(max(omega_min, omega_center - 0.3 * omega_range), min(omega_max, omega_center + 0.3 * omega_range), n_near),
                    np.linspace(min(omega_max, omega_center + 0.3 * omega_range), omega_max, n_far - n_far // 2)
                ])
                omega_samples = np.unique(np.clip(omega_samples, omega_min, omega_max))
        
        # Sample velocities
        for v in v_samples:
            for omega in omega_samples:
                # Simulate trajectory
                trajectory = self._simulate_trajectory(
                    robot_x, robot_y, robot_theta, v, omega, self.lookahead
                )
                
                # Check if trajectory is safe (including dynamic obstacles)
                is_safe = self._check_trajectory_safety(trajectory, lidar_ranges, min_clearance, actor_polygons)
                
                if not is_safe:
                    continue
                
                # Score trajectory
                score = self._score_trajectory(
                    trajectory, goal_x, goal_y, min_clearance, v, omega, is_stuck, is_emergency
                )
                
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_omega = omega
        
        # FIXED: Light smoothing - don't over-smooth and prevent path following
        # Only apply light smoothing if not in emergency or stuck
        if is_emergency or is_stuck:
            # No smoothing when stuck/emergency - need immediate response
            self.last_v = best_v
            self.last_omega = best_omega
        else:
            # Light smoothing for normal operation
            self.last_v = self.alpha * self.last_v + (1 - self.alpha) * best_v
            self.last_omega = self.alpha * self.last_omega + (1 - self.alpha) * best_omega
        
        return self.last_v, self.last_omega
    
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
    
    def _check_trajectory_safety(self, trajectory, lidar_ranges, min_clearance, actor_polygons=None):
        """Check if trajectory is safe (no collisions with static or dynamic obstacles)."""
        # IMPROVED: More aggressive safety checking
        # Require more clearance - ensure trajectory is safe
        safety_margin = 0.9  # Require 90% of min clearance for safety
        if min_clearance < self.min_clearance * safety_margin:
            return False
        
        # Check multiple points along trajectory for collisions with static obstacles
        if self.world is not None and len(trajectory) > 1:
            # Sample every few points along trajectory
            check_indices = np.linspace(0, len(trajectory) - 1, min(5, len(trajectory)), dtype=int)
            for idx in check_indices:
                x, y = trajectory[idx][:2]
                if not self.world.is_free(x, y, use_inflated=True):
                    return False
        
        # IMPROVED: Check for collisions with moving actors (dynamic obstacles)
        if actor_polygons is not None and len(actor_polygons) > 0 and len(trajectory) > 1:
            from shapely.geometry import Point
            # Check trajectory points against moving actor footprints
            check_indices = np.linspace(0, len(trajectory) - 1, min(5, len(trajectory)), dtype=int)
            for idx in check_indices:
                x, y = trajectory[idx][:2]
                traj_point = Point(x, y)
                # Check if any moving actor intersects with trajectory point
                for actor_poly in actor_polygons:
                    if actor_poly.contains(traj_point) or actor_poly.distance(traj_point) < 0.2:
                        # Too close to moving obstacle - unsafe
                        return False
        
        return True
    
    def _score_trajectory(self, trajectory, goal_x, goal_y, min_clearance, v, omega, is_stuck=False, is_emergency=False):
        """
        Score trajectory: w_goal*progress - w_obs*(1/min_clear) - w_smooth*Δu.
        
        Args:
            trajectory: Simulated trajectory (list of (x, y) or (x, y, theta) tuples)
            goal_x, goal_y: Goal waypoint
            min_clearance: Minimum clearance
            v, omega: Velocity command
            is_stuck: Whether robot is stuck (needs backward movement)
            is_emergency: Whether robot is in emergency (dangerously close to obstacle)
        
        Returns:
            Trajectory score
        """
        # IMPROVED: Much more aggressive stuck recovery
        if is_emergency:
            # EMERGENCY MODE: Extremely strong preference for backward movement
            if v < -0.1:  # Moving backward
                backward_bonus = 200.0 * (abs(v) / self.v_max)  # HUGE bonus for backward (was 80.0)
            elif v > 0.1:  # Moving forward in emergency - BAD
                backward_bonus = -100.0  # Huge penalty (was -40.0)
            else:  # No movement in emergency
                backward_bonus = -50.0  # Large penalty (was -10.0)
        elif is_stuck:
            if v < -0.1:  # Moving backward
                # Strong bonus for backward movement when stuck
                backward_bonus = 120.0 * (abs(v) / self.v_max)  # Increased bonus (was 80.0)
            elif v > 0.1:  # Moving forward when stuck
                # Strong penalty - don't move forward into obstacles
                backward_bonus = -60.0  # Increased penalty (was -40.0)
            else:  # No movement
                # Penalty - encourage movement
                backward_bonus = -20.0  # Increased penalty (was -10.0)
        else:
            backward_bonus = 0.0
        # FIXED: Better goal progress calculation with heading alignment bonus
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
            
            # BONUS: Reward alignment with goal direction
            # Check if trajectory is heading toward goal
            if len(trajectory) > 1:
                traj_end = trajectory[-1]
                if len(traj_end) > 2:
                    traj_theta = traj_end[2]
                    # Desired heading to goal
                    dx = goal_x - end_pos[0]
                    dy = goal_y - end_pos[1]
                    if dx*dx + dy*dy > 0.01:
                        desired_theta = np.arctan2(dy, dx)
                        angle_diff = abs(traj_theta - desired_theta)
                        # Normalize angle difference
                        while angle_diff > np.pi:
                            angle_diff -= 2 * np.pi
                        angle_diff = abs(angle_diff)
                        # Bonus for heading toward goal
                        if angle_diff < np.pi / 6:  # Within 30 degrees
                            progress += 0.5  # Significant bonus for good alignment
                        elif angle_diff < np.pi / 3:  # Within 60 degrees
                            progress += 0.2  # Moderate bonus
        else:
            progress = 0.0
        
        # IMPROVED: Much more aggressive obstacle cost - extremely penalize close obstacles
        # Use exponential function to strongly penalize close obstacles
        if min_clearance > 0:
            if min_clearance < self.emergency_stuck_clearance:
                # CRITICAL: Extremely close - massive exponential penalty
                obstacle_cost = 500.0 * np.exp(-min_clearance * 5.0)  # Huge exponential penalty
            elif min_clearance < self.min_clearance * 0.8:
                # Very close to obstacle - exponential penalty
                obstacle_cost = 150.0 * np.exp(-min_clearance * 2.5)  # Increased penalty (was 100.0)
            elif min_clearance < self.min_clearance:
                # Close to obstacle - strong penalty
                obstacle_cost = 80.0 / (min_clearance + 0.01)  # Increased (was 60.0)
            else:
                # Safe distance - still penalize but less
                obstacle_cost = 20.0 / (min_clearance + 0.1)  # Slightly increased (was 15.0)
        else:
            obstacle_cost = 2000.0  # Massive penalty for zero clearance (was 1000.0)
        
        # FIXED: Reduced smoothness cost - don't prevent necessary path-following turns
        # Penalize changes in velocity command (normalized)
        dv = abs(v - self.last_v) / (self.v_max + 0.01)  # Normalize by max velocity
        domega = abs(omega - self.last_omega) / (self.omega_max + 0.01)  # Normalize by max angular velocity
        # REDUCED: Less penalty for necessary turns toward goal
        smoothness_cost = 0.5 * (dv + domega)  # Reduced weight (was 2.0)
        
        # BONUS: Reward trajectories that maintain forward speed (but allow necessary turns)
        if v > 0.5 * self.v_max and abs(v - self.last_v) < 0.2 * self.v_max:
            smoothness_bonus = 5.0  # Reduced bonus (was 10.0) - reward maintaining speed
        else:
            smoothness_bonus = 0.0
        
        # CRITICAL FIX: Bonus for maintaining or increasing speed (encourages faster movement)
        speed_bonus = 0.0
        if v > 0.1:  # Moving forward
            # Bonus for maintaining speed or accelerating
            if v >= abs(self.last_v) * 0.9:  # Maintaining or increasing speed
                speed_bonus = 2.0 * (v / self.v_max)  # Reward based on speed fraction
        elif v < -0.1:  # Moving backward (when stuck)
            speed_bonus = 1.0  # Small bonus for backward movement when needed
        
        # OPTIMIZED: Combined score with better weighting for clean, logical movement
        score = (
            self.w_goal * progress -
            self.w_obs * obstacle_cost -
            self.w_smooth * smoothness_cost +
            backward_bonus +  # Add bonus for backward movement when stuck
            speed_bonus +  # Add bonus for maintaining/gaining speed
            smoothness_bonus  # Add bonus for smooth, predictable movement
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
        # IMPROVED: Much more sensitive replanning triggers
        # Re-plan if:
        # 1. No progress (completely blocked)
        # 2. Clearance dangerously low (needs global replan)
        # 3. Been stuck for a while (local avoidance not working)
        # 4. Emergency clearance - immediate replan needed
        progress_stuck = progress < 0.01  # Increased threshold to catch more cases
        clearance_stuck = min_clearance < self.min_clearance * 0.9  # Trigger much earlier (was 0.7)
        emergency_replan = min_clearance < self.emergency_stuck_clearance  # Emergency threshold
        stuck_for_while = self.stuck_counter > self.stuck_threshold * 1.2  # Trigger earlier (was 1.5)
        
        if progress_stuck or clearance_stuck or stuck_for_while or emergency_replan:
            return True
        return False

