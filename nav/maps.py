"""
Map utilities for navigation.

Purpose: Grid utilities, coordinate conversions, shortest-path baseline.

Inputs:
    - World object with occupancy grid
    - Grid coordinates or world coordinates

Outputs:
    - Grid neighbors, coordinate conversions
    - Shortest-path baseline for efficiency metric

Params:
    world: World object with grid representation
"""

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix


class MapUtils:
    """Map utility functions."""
    
    def __init__(self, world):
        """
        Initialize map utilities.
        
        Args:
            world: World object
        """
        self.world = world
    
    def get_neighbors(self, gx, gy, use_inflated=True):
        """
        Get 8-connected neighbors of a grid cell.
        
        Args:
            gx, gy: Grid coordinates
            use_inflated: Use inflated grid or original occupancy grid
        
        Returns:
            List of (gx, gy) tuples of valid neighbors
        """
        grid = self.world.inflated_grid if use_inflated else self.world.occupancy_grid
        neighbors = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx = gx + dx
                ny = gy + dy
                
                if 0 <= nx < self.world.grid_width and 0 <= ny < self.world.grid_height:
                    if grid[ny, nx] == 0:  # Free space
                        neighbors.append((nx, ny))
        
        return neighbors
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates."""
        return self.world.world_to_grid(x, y)
    
    def grid_to_world(self, gx, gy):
        """Convert grid coordinates to world coordinates."""
        return self.world.grid_to_world(gx, gy)
    
    def compute_shortest_path_baseline(self, start, goal):
        """
        Compute shortest path baseline using Dijkstra on free grid.
        
        Args:
            start: (x, y) world coordinates
            goal: (x, y) world coordinates
        
        Returns:
            float: Shortest path length
        """
        return self.world.compute_shortest_path_baseline(start, goal)
    
    def is_free(self, x, y, use_inflated=True):
        """Check if world position is free."""
        return self.world.is_free(x, y, use_inflated)

