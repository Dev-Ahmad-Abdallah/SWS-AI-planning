"""
BFS (Breadth-First Search) global planner.

Purpose: Grid-based BFS search on inflated occupancy grid.

Inputs:
    - Inflated occupancy grid
    - Start and goal grid coordinates

Outputs:
    - Path as list of (gx, gy) grid coordinates
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional


class BFSPlanner:
    """BFS path planner for grid-based navigation."""
    
    def __init__(self, occupancy_grid, grid_resolution=0.1):
        """
        Initialize BFS planner.
        
        Args:
            occupancy_grid: Inflated occupancy grid (numpy array)
            grid_resolution: Meters per grid cell
        """
        self.grid = occupancy_grid
        self.grid_height, self.grid_width = occupancy_grid.shape
        self.grid_resolution = grid_resolution
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Plan path using BFS.
        
        Args:
            start: (gx, gy) start grid coordinates
            goal: (gx, gy) goal grid coordinates
        
        Returns:
            Path as list of (gx, gy) tuples, or None if no path found
        """
        sx, sy = start
        gx, gy = goal
        
        # Check if start and goal are valid
        if (sx < 0 or sx >= self.grid_width or sy < 0 or sy >= self.grid_height):
            return None
        if (gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height):
            return None
        if self.grid[sy, sx] == 1 or self.grid[gy, gx] == 1:
            return None
        
        # BFS search
        queue = deque([(sx, sy)])
        visited = {(sx, sy)}
        came_from = {(sx, sy): None}
        
        # 8-connected neighbors
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        while queue:
            cx, cy = queue.popleft()
            
            if cx == gx and cy == gy:
                # Reconstruct path
                path = []
                node = (gx, gy)
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path
            
            # Explore neighbors
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                
                if (nx < 0 or nx >= self.grid_width or ny < 0 or ny >= self.grid_height):
                    continue
                if self.grid[ny, nx] == 1:
                    continue
                if (nx, ny) in visited:
                    continue
                
                visited.add((nx, ny))
                came_from[(nx, ny)] = (cx, cy)
                queue.append((nx, ny))
        
        return None

