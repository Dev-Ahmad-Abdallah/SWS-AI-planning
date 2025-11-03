"""
Spanning Tree planners (MST, DFS Spanning Tree, BFS Spanning Tree).

Purpose: Build spanning trees on free space for path planning.

Inputs:
    - Occupancy grid
    - Start and goal grid coordinates

Outputs:
    - Path as list of (gx, gy) grid coordinates
"""

import numpy as np
import heapq
from collections import deque
from typing import List, Tuple, Optional, Set


class MSTPlanner:
    """Minimum Spanning Tree planner using Prim's algorithm."""
    
    def __init__(self, occupancy_grid, grid_resolution=0.1):
        self.grid = occupancy_grid
        self.grid_height, self.grid_width = occupancy_grid.shape
        self.grid_resolution = grid_resolution
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Plan path using MST (finds tree, then shortest path in tree)."""
        # Build MST of free space, then find path from start to goal
        return self._mst_path(start, goal)
    
    def _mst_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Build MST and find path."""
        # Get all free cells
        free_cells = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y, x] == 0:
                    free_cells.append((x, y))
        
        if not free_cells or start not in free_cells or goal not in free_cells:
            return None
        
        # Build MST using Prim's algorithm
        mst = self._prim_mst(free_cells)
        
        # Find path from start to goal in MST using DFS
        path = self._dfs_in_tree(mst, start, goal)
        return path
    
    def _prim_mst(self, nodes: List[Tuple[int, int]]) -> dict:
        """Build MST using Prim's algorithm."""
        if not nodes:
            return {}
        
        mst = {}
        visited = {nodes[0]}
        edges = []
        
        # Initialize priority queue with edges from first node
        for node in nodes[1:]:
            dist = np.sqrt((node[0] - nodes[0][0])**2 + (node[1] - nodes[0][1])**2)
            heapq.heappush(edges, (dist, nodes[0], node))
        
        while edges and len(visited) < len(nodes):
            dist, u, v = heapq.heappop(edges)
            
            if v in visited:
                continue
            
            visited.add(v)
            if u not in mst:
                mst[u] = []
            if v not in mst:
                mst[v] = []
            mst[u].append(v)
            mst[v].append(u)
            
            # Add edges from v
            for node in nodes:
                if node not in visited:
                    dist = np.sqrt((node[0] - v[0])**2 + (node[1] - v[1])**2)
                    heapq.heappush(edges, (dist, v, node))
        
        return mst
    
    def _dfs_in_tree(self, tree: dict, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """DFS to find path in tree."""
        visited = set()
        path = []
        
        def dfs(node):
            if node == goal:
                path.append(node)
                return True
            if node in visited:
                return False
            
            visited.add(node)
            path.append(node)
            
            if node in tree:
                for neighbor in tree[node]:
                    if dfs(neighbor):
                        return True
            
            path.pop()
            return False
        
        if dfs(start):
            return path
        return None


class DFSSpanningTreePlanner:
    """DFS-based spanning tree planner."""
    
    def __init__(self, occupancy_grid, grid_resolution=0.1):
        self.grid = occupancy_grid
        self.grid_height, self.grid_width = occupancy_grid.shape
        self.grid_resolution = grid_resolution
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Build DFS spanning tree and find path - ITERATIVE."""
        tree = {}
        visited = set()
        stack = [(start, None)]  # (node, parent)
        
        # Build tree iteratively
        while stack:
            node, parent = stack.pop()
            
            if node in visited:
                continue
            
            visited.add(node)
            if parent is not None:
                if parent not in tree:
                    tree[parent] = []
                tree[parent].append(node)
            
            # Explore neighbors (reverse order for DFS)
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = node[0] + dx, node[1] + dy
                    neighbor = (nx, ny)
                    if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and
                        self.grid[ny, nx] == 0 and neighbor not in visited):
                        neighbors.append(neighbor)
            
            # Add neighbors to stack in reverse order
            for neighbor in reversed(neighbors):
                stack.append((neighbor, node))
        
        # Find path in tree using iterative DFS
        return self._dfs_in_tree_iterative(tree, start, goal)
    
    def _dfs_in_tree_iterative(self, tree: dict, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Iterative DFS to find path in tree."""
        if start == goal:
            return [start]
        
        stack = [(start, [start])]  # (node, path_to_here)
        visited = set()
        
        while stack:
            node, path_to_here = stack.pop()
            
            if node == goal:
                return path_to_here
            
            if node in visited:
                continue
            
            visited.add(node)
            
            if node in tree:
                # Add neighbors to stack
                for neighbor in reversed(tree[node]):
                    if neighbor not in visited:
                        stack.append((neighbor, path_to_here + [neighbor]))
        
        return None
    
    def _dfs_in_tree(self, tree: dict, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Iterative DFS to find path in tree."""
        if start == goal:
            return [start]
        
        stack = [(start, [start])]  # (node, path_to_here)
        visited = set()
        
        while stack:
            node, path_to_here = stack.pop()
            
            if node == goal:
                return path_to_here
            
            if node in visited:
                continue
            
            visited.add(node)
            
            if node in tree:
                # Add neighbors to stack
                for neighbor in reversed(tree[node]):
                    if neighbor not in visited:
                        stack.append((neighbor, path_to_here + [neighbor]))
        
        return None


class BFSSpanningTreePlanner:
    """BFS-based spanning tree planner."""
    
    def __init__(self, occupancy_grid, grid_resolution=0.1):
        self.grid = occupancy_grid
        self.grid_height, self.grid_width = occupancy_grid.shape
        self.grid_resolution = grid_resolution
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Build BFS spanning tree and find path."""
        tree = {}
        queue = deque([start])
        visited = {start}
        
        while queue:
            node = queue.popleft()
            
            # Explore neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = node[0] + dx, node[1] + dy
                    neighbor = (nx, ny)
                    
                    if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and
                        self.grid[ny, nx] == 0 and neighbor not in visited):
                        visited.add(neighbor)
                        if node not in tree:
                            tree[node] = []
                        tree[node].append(neighbor)
                        queue.append(neighbor)
        
        # Find path in tree using DFS
        return self._dfs_in_tree(tree, start, goal)
    
    def _dfs_in_tree(self, tree: dict, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Iterative DFS to find path in tree."""
        if start == goal:
            return [start]
        
        stack = [(start, [start])]  # (node, path_to_here)
        visited = set()
        
        while stack:
            node, path_to_here = stack.pop()
            
            if node == goal:
                return path_to_here
            
            if node in visited:
                continue
            
            visited.add(node)
            
            if node in tree:
                # Add neighbors to stack
                for neighbor in reversed(tree[node]):
                    if neighbor not in visited:
                        stack.append((neighbor, path_to_here + [neighbor]))
        
        return None


class MaximumSpanningTreePlanner:
    """Maximum Spanning Tree planner (same as MST but maximize edge weights)."""
    
    def __init__(self, occupancy_grid, grid_resolution=0.1):
        self.grid = occupancy_grid
        self.grid_height, self.grid_width = occupancy_grid.shape
        self.grid_resolution = grid_resolution
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Build maximum spanning tree and find path."""
        free_cells = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y, x] == 0:
                    free_cells.append((x, y))
        
        if not free_cells or start not in free_cells or goal not in free_cells:
            return None
        
        # Build maximum spanning tree (negate weights for Prim's)
        mst = self._prim_maxst(free_cells)
        path = self._dfs_in_tree(mst, start, goal)
        return path
    
    def _prim_maxst(self, nodes: List[Tuple[int, int]]) -> dict:
        """Build maximum spanning tree."""
        if not nodes:
            return {}
        
        mst = {}
        visited = {nodes[0]}
        edges = []
        
        for node in nodes[1:]:
            dist = np.sqrt((node[0] - nodes[0][0])**2 + (node[1] - nodes[0][1])**2)
            heapq.heappush(edges, (-dist, nodes[0], node))  # Negate for max
        
        while edges and len(visited) < len(nodes):
            neg_dist, u, v = heapq.heappop(edges)
            
            if v in visited:
                continue
            
            visited.add(v)
            if u not in mst:
                mst[u] = []
            if v not in mst:
                mst[v] = []
            mst[u].append(v)
            mst[v].append(u)
            
            for node in nodes:
                if node not in visited:
                    dist = np.sqrt((node[0] - v[0])**2 + (node[1] - v[1])**2)
                    heapq.heappush(edges, (-dist, v, node))
        
        return mst
    
    def _dfs_in_tree(self, tree: dict, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Iterative DFS to find path in tree."""
        if start == goal:
            return [start]
        
        stack = [(start, [start])]  # (node, path_to_here)
        visited = set()
        
        while stack:
            node, path_to_here = stack.pop()
            
            if node == goal:
                return path_to_here
            
            if node in visited:
                continue
            
            visited.add(node)
            
            if node in tree:
                # Add neighbors to stack
                for neighbor in reversed(tree[node]):
                    if neighbor not in visited:
                        stack.append((neighbor, path_to_here + [neighbor]))
        
        return None


class RootedSpanningTreePlanner:
    """Rooted spanning tree planner (rooted at start)."""
    
    def __init__(self, occupancy_grid, grid_resolution=0.1):
        self.grid = occupancy_grid
        self.grid_height, self.grid_width = occupancy_grid.shape
        self.grid_resolution = grid_resolution
    
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Build rooted spanning tree and find path."""
        tree = {}
        queue = deque([start])
        visited = {start}
        
        while queue:
            node = queue.popleft()
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = node[0] + dx, node[1] + dy
                    neighbor = (nx, ny)
                    
                    if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and
                        self.grid[ny, nx] == 0 and neighbor not in visited):
                        visited.add(neighbor)
                        if node not in tree:
                            tree[node] = []
                        tree[node].append(neighbor)
                        queue.append(neighbor)
        
        # Path from root (start) to goal
        return self._dfs_in_tree(tree, start, goal)
    
    def _dfs_in_tree(self, tree: dict, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Iterative DFS to find path in tree."""
        if start == goal:
            return [start]
        
        stack = [(start, [start])]  # (node, path_to_here)
        visited = set()
        
        while stack:
            node, path_to_here = stack.pop()
            
            if node == goal:
                return path_to_here
            
            if node in visited:
                continue
            
            visited.add(node)
            
            if node in tree:
                # Add neighbors to stack
                for neighbor in reversed(tree[node]):
                    if neighbor not in visited:
                        stack.append((neighbor, path_to_here + [neighbor]))
        
        return None

