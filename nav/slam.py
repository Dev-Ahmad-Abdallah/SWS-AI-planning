"""
SLAM (Simultaneous Localization and Mapping) implementation.

Purpose: Real-time mapping and localization for robot exploration.

Inputs:
    - LiDAR scans
    - Robot odometry
    - Current position estimate

Outputs:
    - Occupancy grid map
    - Robot pose estimate
"""

import numpy as np
from typing import Tuple, Optional
from collections import deque


class SLAM:
    """Simple SLAM implementation using LiDAR and odometry."""
    
    def __init__(self, world_width, world_height, grid_resolution=0.1,
                 max_range=8.0, hit_prob=0.7, miss_prob=0.3):
        """
        Initialize SLAM.
        
        Args:
            world_width, world_height: World dimensions
            grid_resolution: Grid resolution
            max_range: Maximum LiDAR range
            hit_prob: Probability of hit (occupied)
            miss_prob: Probability of miss (free)
        """
        self.world_width = world_width
        self.world_height = world_height
        self.grid_resolution = grid_resolution
        self.grid_width = int(world_width / grid_resolution)
        self.grid_height = int(world_height / grid_resolution)
        
        # Occupancy grid (log-odds representation)
        self.log_odds = np.zeros((self.grid_height, self.grid_width))
        
        # Robot pose estimate (x, y, theta)
        self.pose_estimate = None
        
        # SLAM parameters
        self.max_range = max_range
        self.hit_prob = hit_prob
        self.miss_prob = miss_prob
        
        # Log-odds thresholds
        self.lo_occupied = np.log(hit_prob / (1 - hit_prob))
        self.lo_free = np.log(miss_prob / (1 - miss_prob))
        self.lo_max = 5.0
        self.lo_min = -5.0
        
        # Pose history for odometry integration
        self.pose_history = deque(maxlen=100)
    
    def update(self, robot_x, robot_y, robot_theta, lidar_ranges, lidar_angles):
        """
        Update SLAM with new sensor data.
        
        Args:
            robot_x, robot_y, robot_theta: Robot pose from odometry
            lidar_ranges: Array of range measurements
            lidar_angles: Array of beam angles (relative to robot heading)
        """
        # Update pose estimate (simple odometry for now)
        if self.pose_estimate is None:
            self.pose_estimate = (robot_x, robot_y, robot_theta)
        else:
            # Simple pose estimation (could be improved with EKF)
            self.pose_estimate = (robot_x, robot_y, robot_theta)
        
        self.pose_history.append(self.pose_estimate)
        
        # Update occupancy grid with LiDAR scan
        self._update_map(robot_x, robot_y, robot_theta, lidar_ranges, lidar_angles)
    
    def _update_map(self, robot_x, robot_y, robot_theta, ranges, angles):
        """Update occupancy grid with LiDAR measurements."""
        robot_gx, robot_gy = self.world_to_grid(robot_x, robot_y)
        
        for i, (range_val, angle) in enumerate(zip(ranges, angles)):
            if range_val >= self.max_range:
                continue
            
            # Beam angle in world frame
            beam_angle = robot_theta + angle
            
            # Endpoint of beam
            end_x = robot_x + range_val * np.cos(beam_angle)
            end_y = robot_y + range_val * np.sin(beam_angle)
            
            # Update cells along the ray
            self._update_ray(robot_gx, robot_gy, end_x, end_y, range_val)
    
    def _update_ray(self, start_gx, start_gy, end_x, end_y, range_val):
        """Update log-odds along a ray using Bresenham line algorithm."""
        end_gx, end_gy = self.world_to_grid(end_x, end_y)
        
        # Bresenham line algorithm
        dx = abs(end_gx - start_gx)
        dy = abs(end_gy - start_gy)
        sx = 1 if start_gx < end_gx else -1
        sy = 1 if start_gy < end_gy else -1
        err = dx - dy
        
        x, y = start_gx, start_gy
        
        while True:
            # Update cell as free (unless it's the endpoint)
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                if (x, y) == (end_gx, end_gy):
                    # Endpoint is occupied (hit)
                    self.log_odds[y, x] = np.clip(
                        self.log_odds[y, x] + self.lo_occupied,
                        self.lo_min, self.lo_max
                    )
                else:
                    # Ray cells are free (miss)
                    self.log_odds[y, x] = np.clip(
                        self.log_odds[y, x] + self.lo_free,
                        self.lo_min, self.lo_max
                    )
            
            if x == end_gx and y == end_gy:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def get_map(self) -> np.ndarray:
        """
        Get occupancy grid from log-odds.
        
        Returns:
            Occupancy grid (0=free, 1=occupied, -1=unknown)
        """
        # Convert log-odds to probabilities
        prob = 1.0 / (1.0 + np.exp(-self.log_odds))
        
        # Threshold to occupancy grid
        grid = np.zeros_like(self.log_odds, dtype=int)
        grid[prob > 0.5] = 1  # Occupied
        grid[prob < 0.3] = 0  # Free
        grid[(prob >= 0.3) & (prob <= 0.5)] = -1  # Unknown
        
        return grid
    
    def world_to_grid(self, wx, wy) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        gx = int(wx / self.grid_resolution)
        gy = int(wy / self.grid_resolution)
        return gx, gy
    
    def grid_to_world(self, gx, gy) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        wx = gx * self.grid_resolution
        wy = gy * self.grid_resolution
        return wx, wy
    
    def get_pose(self) -> Optional[Tuple[float, float, float]]:
        """Get current pose estimate."""
        return self.pose_estimate

