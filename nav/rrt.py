"""
RRT (Rapidly-exploring Random Tree) global planner.

Purpose: Sampling-based path planning using RRT.

Inputs:
    - World dimensions
    - Occupancy grid
    - Start and goal world coordinates

Outputs:
    - Path as list of (x, y) world coordinates
"""

import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from typing import List, Tuple, Optional


class RRTPlanner:
    """RRT path planner."""
    
    def __init__(self, world_width, world_height, occupancy_grid, grid_resolution=0.1,
                 max_iterations=5000, step_size=0.5, goal_bias=0.1):
        """
        Initialize RRT planner.
        
        Args:
            world_width, world_height: World dimensions
            occupancy_grid: Occupancy grid (0=free, 1=obstacle)
            grid_resolution: Grid resolution
            max_iterations: Maximum iterations
            step_size: Step size for tree expansion
            goal_bias: Probability of sampling goal (0-1)
        """
        self.world_width = world_width
        self.world_height = world_height
        self.grid = occupancy_grid
        self.grid_height, self.grid_width = occupancy_grid.shape
        self.grid_resolution = grid_resolution
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        
        # Build obstacle polygons for collision checking
        self._build_obstacles()
    
    def _build_obstacles(self):
        """Build obstacle polygons from grid."""
        from shapely.geometry import box
        
        obstacles = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Only treat walls (1) and shelves (2) as obstacles - aisles (3), dock (4), staging (5) are free
                # Since we receive inflated grid, check if value is 1 (obstacle)
                if self.grid[y, x] == 1:
                    wx = x * self.grid_resolution
                    wy = y * self.grid_resolution
                    cell = box(
                        wx, wy,
                        wx + self.grid_resolution,
                        wy + self.grid_resolution
                    )
                    obstacles.append(cell)
        
        if obstacles:
            self.obstacle_union = unary_union(obstacles)
        else:
            self.obstacle_union = None
    
    def _is_free(self, point: Tuple[float, float]) -> bool:
        """Check if point is in free space."""
        px, py = point
        if px < 0 or px >= self.world_width or py < 0 or py >= self.world_height:
            return False
        
        p = Point(px, py)
        if self.obstacle_union is not None:
            return not self.obstacle_union.intersects(p)
        return True
    
    def _is_path_free(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Check if path between two points is collision-free."""
        line = LineString([p1, p2])
        if self.obstacle_union is not None:
            return not self.obstacle_union.intersects(line)
        return True
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path using RRT.
        
        Args:
            start: (x, y) start world coordinates
            goal: (x, y) goal world coordinates
        
        Returns:
            Path as list of (x, y) tuples, or None if no path found
        """
        if not self._is_free(start) or not self._is_free(goal):
            return None
        
        # RRT tree: node -> parent
        tree = {start: None}
        
        for iteration in range(self.max_iterations):
            # Sample random point (with goal bias)
            if np.random.random() < self.goal_bias:
                rand_point = goal
            else:
                rand_point = (np.random.uniform(0, self.world_width),
                            np.random.uniform(0, self.world_height))
            
            # Find nearest node in tree
            nearest = min(tree.keys(), 
                         key=lambda n: np.sqrt((n[0] - rand_point[0])**2 + (n[1] - rand_point[1])**2))
            
            # Extend towards random point
            direction = np.array([rand_point[0] - nearest[0], rand_point[1] - nearest[1]])
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
            
            new_point = (
                nearest[0] + direction[0] * min(self.step_size, dist),
                nearest[1] + direction[1] * min(self.step_size, dist)
            )
            
            # Check if new point is valid
            if self._is_free(new_point) and self._is_path_free(nearest, new_point):
                tree[new_point] = nearest
                
                # Check if we're close enough to goal
                if np.sqrt((new_point[0] - goal[0])**2 + (new_point[1] - goal[1])**2) < self.step_size:
                    if self._is_path_free(new_point, goal):
                        tree[goal] = new_point
                        # Reconstruct path
                        path = []
                        node = goal
                        while node is not None:
                            path.append(node)
                            node = tree[node]
                        path.reverse()
                        return path
        
        return None

