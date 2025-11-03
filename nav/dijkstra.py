"""
Dijkstra global planner.

Purpose: Grid-based Dijkstra search on inflated occupancy grid.

Inputs:
    - Inflated occupancy grid
    - Start and goal grid coordinates

Outputs:
    - Path as list of (gx, gy) grid coordinates
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional


class DijkstraPlanner:
    """Dijkstra path planner for grid-based navigation."""
    
    def __init__(self, occupancy_grid, grid_resolution=0.1):
        """
        Initialize Dijkstra planner.
        
        Args:
            occupancy_grid: Inflated occupancy grid (numpy array)
            grid_resolution: Meters per grid cell
        """
        self.grid = occupancy_grid
        self.grid_height, self.grid_width = occupancy_grid.shape
        self.grid_resolution = grid_resolution
        self.sqrt2 = np.sqrt(2)
        self.d1 = 1.0 * grid_resolution
        self.d2 = self.sqrt2 * grid_resolution
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Plan path using Dijkstra.
        
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
        
        # Dijkstra search
        open_set = []
        heapq.heappush(open_set, (0.0, sx, sy))
        
        came_from = {}
        g_score = {}
        g_score[(sx, sy)] = 0.0
        visited = set()
        
        while open_set:
            current_g, cx, cy = heapq.heappop(open_set)
            
            if (cx, cy) in visited:
                continue
            
            visited.add((cx, cy))
            
            if cx == gx and cy == gy:
                # Reconstruct path
                path = []
                node = (gx, gy)
                while node is not None:
                    path.append(node)
                    node = came_from.get(node)
                path.reverse()
                return path
            
            # Explore neighbors (8-connected)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx = cx + dx
                    ny = cy + dy
                    
                    if nx < 0 or nx >= self.grid_width or ny < 0 or ny >= self.grid_height:
                        continue
                    if self.grid[ny, nx] == 1:
                        continue
                    if (nx, ny) in visited:
                        continue
                    
                    # Compute edge cost with clearance penalty
                    if abs(dx) + abs(dy) == 2:  # Diagonal
                        base_cost = self.d2
                    else:
                        base_cost = self.d1
                    
                    # CRITICAL FIX: Add penalty for being close to obstacles
                    # Check neighbors to see if close to obstacle
                    clearance_penalty = 0.0
                    for check_dx in [-1, 0, 1]:
                        for check_dy in [-1, 0, 1]:
                            check_x, check_y = nx + check_dx, ny + check_dy
                            if (0 <= check_x < self.grid_width and 0 <= check_y < self.grid_height):
                                if self.grid[check_y, check_x] == 1:  # Obstacle nearby
                                    # Penalize being close to obstacles
                                    clearance_penalty += 0.3 * self.grid_resolution
                    
                    edge_cost = base_cost + clearance_penalty
                    tentative_g = current_g + edge_cost
                    
                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = tentative_g
                        came_from[(nx, ny)] = (cx, cy)
                        heapq.heappush(open_set, (tentative_g, nx, ny))
        
        return None

