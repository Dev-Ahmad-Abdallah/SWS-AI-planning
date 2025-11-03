"""
A* global planner.

Purpose: Grid-based A* search on inflated occupancy grid with octile heuristic.

Inputs:
    - Inflated occupancy grid
    - Start and goal grid coordinates
    - Grid resolution

Outputs:
    - Path as list of (gx, gy) grid coordinates
    - Planning stats (runtime, path length, nodes expanded)

Params:
    occupancy_grid: numpy array - Inflated occupancy grid (0=free, 1=obstacle)
    grid_resolution: float - Meters/pixels per grid cell
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional


class AStarPlanner:
    """A* path planner for grid-based navigation."""
    
    def __init__(self, occupancy_grid, grid_resolution=0.1):
        """
        Initialize A* planner.
        
        Args:
            occupancy_grid: Inflated occupancy grid (numpy array)
            grid_resolution: Meters/pixels per grid cell
        """
        self.grid = occupancy_grid
        self.grid_height, self.grid_width = occupancy_grid.shape
        self.grid_resolution = grid_resolution
        
        # Heuristic weights
        self.sqrt2 = np.sqrt(2)
        self.d1 = 1.0 * grid_resolution  # Cost for horizontal/vertical
        self.d2 = self.sqrt2 * grid_resolution  # Cost for diagonal
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Plan path using A*.
        
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
        
        # A* search
        open_set = []  # Priority queue: (f, g, x, y, parent_x, parent_y)
        heapq.heappush(open_set, (0.0, 0.0, sx, sy, None, None))
        
        came_from = {}  # (x, y) -> (parent_x, parent_y)
        g_score = {}  # (x, y) -> g cost
        g_score[(sx, sy)] = 0.0
        
        nodes_expanded = 0
        
        while open_set:
            current_f, current_g, cx, cy, px, py = heapq.heappop(open_set)
            
            # Skip if we've already found a better path to this node
            if (cx, cy) in g_score and g_score[(cx, cy)] < current_g:
                continue
            
            nodes_expanded += 1
            came_from[(cx, cy)] = (px, py)
            
            # Check if goal reached
            if cx == gx and cy == gy:
                # Reconstruct path
                path = []
                node = (gx, gy)
                while node is not None:
                    path.append(node)
                    parent = came_from.get(node)
                    if parent is None:
                        break
                    # Check if parent is (None, None) which means we're at start
                    if parent[0] is None or parent[1] is None:
                        break
                    node = parent
                path.reverse()
                return path
            
            # Explore neighbors (8-connected)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx = cx + dx
                    ny = cy + dy
                    
                    # Check bounds
                    if nx < 0 or nx >= self.grid_width or ny < 0 or ny >= self.grid_height:
                        continue
                    
                    # Check if free
                    if self.grid[ny, nx] == 1:
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
                    
                    # Check if we've seen this node with better cost
                    if (nx, ny) in g_score and g_score[(nx, ny)] <= tentative_g:
                        continue
                    
                    # Compute heuristic (octile distance)
                    h = self._octile_heuristic(nx, ny, gx, gy)
                    f = tentative_g + h
                    
                    heapq.heappush(open_set, (f, tentative_g, nx, ny, cx, cy))
                    g_score[(nx, ny)] = tentative_g
        
        # No path found
        return None
    
    def _octile_heuristic(self, x1, y1, x2, y2):
        """
        Octile heuristic: max(|dx|, |dy|) + (√2 - 1) * min(|dx|, |dy|).
        
        Args:
            x1, y1: Start grid coordinates
            x2, y2: Goal grid coordinates
        
        Returns:
            Heuristic distance in world units
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        # Octile distance
        if dx > dy:
            return (self.sqrt2 - 1) * dy * self.grid_resolution + dx * self.grid_resolution
        else:
            return (self.sqrt2 - 1) * dx * self.grid_resolution + dy * self.grid_resolution
    
    def smooth_path(self, path):
        """
        Optional path smoothing using line-of-sight.
        
        Args:
            path: List of (gx, gy) tuples
        
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        
        current = 0
        while current < len(path) - 1:
            # Try to skip ahead as far as possible
            for i in range(len(path) - 1, current, -1):
                if self._line_of_sight(path[current], path[i]):
                    smoothed.append(path[i])
                    current = i
                    break
            else:
                # No line of sight, advance one step
                current += 1
                if current < len(path):
                    smoothed.append(path[current])
        
        return smoothed
    
    def _line_of_sight(self, start, end):
        """Check if there's a clear line of sight between two grid cells."""
        sx, sy = start
        ex, ey = end
        
        # Bresenham line algorithm
        dx = abs(ex - sx)
        dy = abs(ey - sy)
        sx_step = 1 if sx < ex else -1
        sy_step = 1 if sy < ey else -1
        err = dx - dy
        
        x, y = sx, sy
        
        while True:
            # Check if cell is free
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                if self.grid[y, x] == 1:
                    return False
            else:
                return False
            
            if x == ex and y == ey:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx_step
            if e2 < dx:
                err += dx
                y += sy_step
        
        return True

