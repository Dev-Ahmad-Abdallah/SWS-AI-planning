"""
PRM (Probabilistic Roadmap) global planner.

Purpose: Sampling-based path planning using PRM on free space.

Inputs:
    - World dimensions
    - Occupancy grid
    - Start and goal world coordinates
    - Number of samples, k neighbors

Outputs:
    - Path as list of (x, y) world coordinates
    - Planning stats (runtime, path length, samples used)

Params:
    world_width, world_height: float - World dimensions
    occupancy_grid: numpy array - Occupancy grid
    grid_resolution: float - Grid resolution
    num_samples: int - Number of PRM samples
    k_neighbors: int - Number of nearest neighbors to connect
"""

import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from typing import List, Tuple, Optional


class PRMPlanner:
    """Probabilistic Roadmap (PRM) planner."""
    
    def __init__(self, world_width, world_height, occupancy_grid, grid_resolution=0.1,
                 num_samples=500, k_neighbors=10):
        """
        Initialize PRM planner.
        
        Args:
            world_width, world_height: World dimensions
            occupancy_grid: Occupancy grid (0=free, 1=obstacle)
            grid_resolution: Grid resolution
            num_samples: Number of PRM samples
            k_neighbors: Number of nearest neighbors to connect
        """
        self.world_width = world_width
        self.world_height = world_height
        self.grid = occupancy_grid
        self.grid_height, self.grid_width = occupancy_grid.shape
        self.grid_resolution = grid_resolution
        self.num_samples = num_samples
        self.k_neighbors = k_neighbors
        
        # Build obstacle polygons
        self.obstacle_polygons = self._build_obstacle_polygons()
        
        # PRM roadmap (will be built on first planning)
        self.roadmap = None
        self.samples = []
    
    def _build_obstacle_polygons(self):
        """Build obstacle polygons from occupancy grid."""
        obstacles = []
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y, x] == 1:
                    # Convert grid cell to world coordinates
                    world_x = x * self.grid_resolution
                    world_y = y * self.grid_resolution
                    # Create rectangle for cell
                    from shapely.geometry import box
                    cell = box(
                        world_x,
                        world_y,
                        world_x + self.grid_resolution,
                        world_y + self.grid_resolution
                    )
                    obstacles.append(cell)
        
        if obstacles:
            return unary_union(obstacles)
        else:
            return None
    
    def _is_free(self, x, y):
        """Check if world position is free."""
        if x < 0 or x >= self.world_width or y < 0 or y >= self.world_height:
            return False
        
        if self.obstacle_polygons is None:
            return True
        
        point = Point(x, y)
        return not self.obstacle_polygons.intersects(point)
    
    def _collision_check_edge(self, start, end):
        """Check if edge between two points is collision-free."""
        line = LineString([start, end])
        
        if self.obstacle_polygons is None:
            return True
        
        return not self.obstacle_polygons.intersects(line)
    
    def _sample_free_space(self, num_samples):
        """Sample free space uniformly."""
        samples = []
        attempts = 0
        max_attempts = num_samples * 10
        
        while len(samples) < num_samples and attempts < max_attempts:
            x = np.random.uniform(0, self.world_width)
            y = np.random.uniform(0, self.world_height)
            
            if self._is_free(x, y):
                samples.append((x, y))
            
            attempts += 1
        
        return samples
    
    def _build_roadmap(self):
        """Build PRM roadmap."""
        # Sample free space
        self.samples = self._sample_free_space(self.num_samples)
        
        if len(self.samples) < 2:
            return False
        
        # Create graph
        self.roadmap = nx.Graph()
        
        # Add nodes
        for i, sample in enumerate(self.samples):
            self.roadmap.add_node(i, pos=sample)
        
        # Connect k nearest neighbors
        for i, sample1 in enumerate(self.samples):
            # Find k nearest neighbors
            distances = []
            for j, sample2 in enumerate(self.samples):
                if i != j:
                    dist = np.sqrt((sample1[0] - sample2[0])**2 + (sample1[1] - sample2[1])**2)
                    distances.append((dist, j))
            
            distances.sort(key=lambda x: x[0])
            
            # Connect to k nearest neighbors (if collision-free)
            for dist, j in distances[:self.k_neighbors]:
                if self._collision_check_edge(sample1, self.samples[j]):
                    self.roadmap.add_edge(i, j, weight=dist)
        
        return True
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path using PRM.
        
        Args:
            start: (x, y) start world coordinates
            goal: (x, y) goal world coordinates
        
        Returns:
            Path as list of (x, y) tuples, or None if no path found
        """
        # Check if start and goal are free
        if not self._is_free(start[0], start[1]) or not self._is_free(goal[0], goal[1]):
            return None
        
        # Build roadmap if not already built
        if self.roadmap is None:
            if not self._build_roadmap():
                return None
        
        # Add start and goal to roadmap
        start_idx = len(self.samples)
        goal_idx = len(self.samples) + 1
        
        self.roadmap.add_node(start_idx, pos=start)
        self.roadmap.add_node(goal_idx, pos=goal)
        
        # Connect start and goal to nearest nodes
        start_distances = []
        goal_distances = []
        
        for i, sample in enumerate(self.samples):
            # Distance to start
            dist_start = np.sqrt((start[0] - sample[0])**2 + (start[1] - sample[1])**2)
            if self._collision_check_edge(start, sample):
                start_distances.append((dist_start, i))
            
            # Distance to goal
            dist_goal = np.sqrt((goal[0] - sample[0])**2 + (goal[1] - sample[1])**2)
            if self._collision_check_edge(goal, sample):
                goal_distances.append((dist_goal, i))
        
        # Connect start to k nearest
        start_distances.sort(key=lambda x: x[0])
        for dist, i in start_distances[:self.k_neighbors]:
            self.roadmap.add_edge(start_idx, i, weight=dist)
        
        # Connect goal to k nearest
        goal_distances.sort(key=lambda x: x[0])
        for dist, i in goal_distances[:self.k_neighbors]:
            self.roadmap.add_edge(goal_idx, i, weight=dist)
        
        # Try to find direct connection
        if self._collision_check_edge(start, goal):
            dist = np.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)
            self.roadmap.add_edge(start_idx, goal_idx, weight=dist)
        
        # Find shortest path using Dijkstra
        try:
            path_nodes = nx.shortest_path(self.roadmap, start_idx, goal_idx, weight='weight')
            
            # Convert node indices to world coordinates
            path = []
            for node_idx in path_nodes:
                if node_idx == start_idx:
                    path.append(start)
                elif node_idx == goal_idx:
                    path.append(goal)
                else:
                    path.append(self.samples[node_idx])
            
            # Remove start and goal nodes from roadmap for next planning
            self.roadmap.remove_node(start_idx)
            self.roadmap.remove_node(goal_idx)
            
            return path
        
        except nx.NetworkXNoPath:
            # Remove start and goal nodes
            self.roadmap.remove_node(start_idx)
            self.roadmap.remove_node(goal_idx)
            return None

