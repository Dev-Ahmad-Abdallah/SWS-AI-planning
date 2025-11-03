"""
World system for warehouse simulation.

Purpose: Load grid/tile map, compute occupancy grid, handle inflation (C-space),
         provide coordinate transforms, compute shortest-path baseline.

Inputs:
    - Map layout (ASCII or programmatic)
    - Robot footprint dimensions
    - Inflation radius from config

Outputs:
    - Occupancy grid (numpy array: 0=free, 1=obstacle)
    - Inflated occupancy grid for planning
    - Shortest-path baseline for efficiency metric
    - Coordinate transform functions

Params:
    width: float - World width in meters/pixels
    height: float - World height in meters/pixels
    grid_resolution: float - Meters/pixels per grid cell
    robot_width: float - Robot footprint width
    robot_length: float - Robot footprint length
    inflation_radius: float - Inflation radius for C-space
"""

import numpy as np
from pathlib import Path
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


class World:
    """Warehouse world with grid-based representation and inflation."""
    
    def __init__(self, width=40.0, height=30.0, grid_resolution=0.1, 
                 robot_width=0.5, robot_length=0.7, inflation_radius=0.3):
        """
        Initialize world.
        
        Args:
            width: World width in meters/pixels
            height: World height in meters/pixels
            grid_resolution: Meters/pixels per grid cell
            robot_width: Robot footprint width
            robot_length: Robot footprint length
            inflation_radius: Inflation radius for C-space
        """
        self.width = width
        self.height = height
        self.grid_resolution = grid_resolution
        self.grid_width = int(width / grid_resolution)
        self.grid_height = int(height / grid_resolution)
        
        self.robot_width = robot_width
        self.robot_length = robot_length
        self.inflation_radius = inflation_radius
        
        # Create warehouse layout programmatically
        self.occupancy_grid = self._create_warehouse_layout()
        
        # Compute inflated grid for C-space
        self.inflated_grid = self._compute_inflation()
        
        # Compute shortest-path baseline (for efficiency metric)
        self.shortest_path_baseline = None
        
    def _create_warehouse_layout(self):
        """
        Create realistic warehouse layout with aisles, shelves, dock.
        Grid encoding: 0=free, 1=wall, 2=shelf, 3=aisle, 4=dock, 5=staging
        
        Returns:
            numpy array: 0=free, 1-5=obstacle types
        """
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # Create walls around perimeter (type 1 = wall)
        grid[0, :] = 1  # top wall
        grid[-1, :] = 1  # bottom wall
        grid[:, 0] = 1  # left wall
        grid[:, -1] = 1  # right wall
        
        # Realistic warehouse dimensions
        # Aisle width: ~3-4m (for forklifts), shelf depth: ~1.2m, shelf height: ~8-10m
        # Note: aisle_width_cells will be zone-specific (defined below)
        shelf_depth_cells = int(1.2 / self.grid_resolution)
        shelf_height_cells = int(8.0 / self.grid_resolution)
        
        # Create organized storage zones with multiple aisles
        # Zone 1: High-density storage (left side) - more shelves, narrower aisles
        # Zone 2: Medium storage (middle) - standard layout
        # Zone 3: Fast-moving goods (right side) - wider aisles for faster access
        
        num_zones = 3
        zone_width = self.grid_width // num_zones
        
        # CRITICAL FIX: Make crowded scenarios more realistic by varying aisle widths
        # Different zones have different aisle widths for more realistic warehouse layout
        zone_aisle_widths = [
            int(3.0 / self.grid_resolution),  # Zone 1: Narrower (3m) - more crowded
            int(3.5 / self.grid_resolution),  # Zone 2: Standard (3.5m)
            int(4.0 / self.grid_resolution),   # Zone 3: Wider (4m) - less crowded
        ]
        
        for zone in range(num_zones):
            zone_start_x = zone * zone_width
            zone_end_x = (zone + 1) * zone_width
            
            # Number of aisles per zone (3-4 aisles)
            num_aisles = 4 if zone == 1 else 3  # Middle zone has more aisles
            
            # Use zone-specific aisle width
            current_aisle_width_cells = zone_aisle_widths[zone]
            
            for aisle_num in range(num_aisles):
                # Calculate aisle position within zone
                aisle_spacing = (zone_end_x - zone_start_x) / (num_aisles + 1)
                aisle_x = int(zone_start_x + (aisle_num + 1) * aisle_spacing)
                
                # REALISTIC WAREHOUSE: Create shelf rows on left side of aisle (type 2 = shelf)
                # CRITICAL FIX: Make shelves more realistic - variable heights and gaps
                shelf_left = max(1, aisle_x - shelf_depth_cells - current_aisle_width_cells // 2)
                shelf_right = max(1, aisle_x - current_aisle_width_cells // 2)
                shelf_top = int(0.20 * self.grid_height)  # Leave space for top cross-corridor
                shelf_bottom = int(0.78 * self.grid_height)  # Leave space for dock area
                
                # Create more realistic shelves with gaps for cross-aisles and breaks
                if shelf_bottom > shelf_top and shelf_right > shelf_left:
                    # Add vertical gaps every few meters for cross-aisles (more realistic)
                    gap_spacing = int(8.0 / self.grid_resolution)  # Gap every ~8m
                    for shelf_y in range(shelf_top, shelf_bottom, gap_spacing + int(2.0 / self.grid_resolution)):
                        shelf_end_y = min(shelf_bottom, shelf_y + int(3.0 / self.grid_resolution))
                        if shelf_end_y > shelf_y:
                            grid[shelf_y:shelf_end_y, shelf_left:shelf_right] = 2  # Shelf type
                    
                    # Also add some breaks in shelves for more realistic warehouse layout
                    if aisle_num % 2 == 0:  # Alternate aisles have breaks
                        break_y = shelf_top + int((shelf_bottom - shelf_top) * 0.3)
                        break_height = int(2.0 / self.grid_resolution)
                        grid[break_y:break_y+break_height, shelf_left:shelf_right] = 0  # Break in shelf
                
                # REALISTIC WAREHOUSE: Create shelf rows on right side of aisle
                shelf_left = min(self.grid_width - 1, aisle_x + current_aisle_width_cells // 2)
                shelf_right = min(self.grid_width - 1, aisle_x + current_aisle_width_cells // 2 + shelf_depth_cells)
                # Only place shelves if there's room
                if shelf_bottom > shelf_top and shelf_right > shelf_left:
                    # Add vertical gaps every few meters for cross-aisles (more realistic)
                    gap_spacing = int(8.0 / self.grid_resolution)  # Gap every ~8m
                    for shelf_y in range(shelf_top, shelf_bottom, gap_spacing + int(2.0 / self.grid_resolution)):
                        shelf_end_y = min(shelf_bottom, shelf_y + int(3.0 / self.grid_resolution))
                        if shelf_end_y > shelf_y:
                            grid[shelf_y:shelf_end_y, shelf_left:shelf_right] = 2  # Shelf type
                    
                    # Also add some breaks in shelves for more realistic warehouse layout
                    if aisle_num % 2 == 1:  # Alternate aisles have breaks on opposite side
                        break_y = shelf_top + int((shelf_bottom - shelf_top) * 0.5)
                        break_height = int(2.0 / self.grid_resolution)
                        grid[break_y:break_y+break_height, shelf_left:shelf_right] = 0  # Break in shelf
                
                # CRITICAL: Ensure aisle itself is FREE (type 3 = aisle) - walkable path between shelves
                aisle_left = max(0, aisle_x - current_aisle_width_cells // 2)
                aisle_right = min(self.grid_width, aisle_x + current_aisle_width_cells // 2)
                # Force aisle to be free (overwrite any shelves)
                for y in range(shelf_top, shelf_bottom):
                    for x in range(aisle_left, aisle_right):
                        grid[y, x] = 3  # Force aisle type - GUARANTEED WALKABLE PATH
        
        # REALISTIC WAREHOUSE: Main cross-corridors (horizontal aisles) - CRITICAL PATHS
        # Top cross-corridor for access to storage zones - GUARANTEED FREE
        corridor_y = int(0.10 * self.grid_height)
        corridor_height = int(3.0 / self.grid_resolution)  # Wider corridor
        for y in range(max(0, corridor_y), min(corridor_y+corridor_height, self.grid_height)):
            for x in range(self.grid_width):
                grid[y, x] = 3  # Force aisle type - OVERWRITE shelves if needed
        
        # Bottom cross-corridor near dock for loading/unloading access - GUARANTEED FREE
        corridor_y = int(0.80 * self.grid_height)
        corridor_height = int(3.0 / self.grid_resolution)
        for y in range(max(0, corridor_y), min(corridor_y+corridor_height, self.grid_height)):
            for x in range(self.grid_width):
                grid[y, x] = 3  # Force aisle type - OVERWRITE shelves if needed
        
        # REALISTIC WAREHOUSE: Vertical main corridor (center spine) - CRITICAL PATH
        # This is the main navigation corridor - make it WIDE and GUARANTEED FREE
        corridor_x = int(0.5 * self.grid_width)
        corridor_width = int(6.0 / self.grid_resolution)  # WIDE main corridor for forklifts
        # Clear the entire corridor first (remove shelves if they overlap)
        for y in range(self.grid_height):
            for x in range(max(0, corridor_x-corridor_width//2), min(corridor_x+corridor_width//2, self.grid_width)):
                grid[y, x] = 3  # Force aisle type - OVERWRITE shelves if needed
        
        # REALISTIC WAREHOUSE: Dock area (bottom center, wide for loading/unloading) - type 4 = dock
        # Dock should be FREE space, not an obstacle
        dock_y_start = int(0.85 * self.grid_height)
        dock_y_end = int(0.98 * self.grid_height)
        dock_width = int(0.6 * self.grid_width)  # Large dock area
        dock_start_x = int(0.20 * self.grid_width)
        dock_end_x = dock_start_x + dock_width
        # Mark dock as aisle (free space) with visual marker
        for y in range(dock_y_start, min(dock_y_end, self.grid_height)):
            for x in range(dock_start_x, min(dock_end_x, self.grid_width)):
                if grid[y, x] == 0:  # Only mark free space
                    grid[y, x] = 4  # Dock type (visually marked but navigable)
        
        # REALISTIC WAREHOUSE: Receiving/Staging area (top left) - type 5 = staging
        # Staging should be FREE space, not an obstacle
        staging_y = int(0.02 * self.grid_height)
        staging_x = int(0.02 * self.grid_width)
        staging_width = int(0.20 * self.grid_width)
        staging_height = int(0.10 * self.grid_height)
        # Mark staging as aisle (free space) with visual marker
        for y in range(staging_y, min(staging_y+staging_height, self.grid_height)):
            for x in range(staging_x, min(staging_x+staging_width, self.grid_width)):
                if grid[y, x] == 0:  # Only mark free space
                    grid[y, x] = 5  # Staging type (visually marked but navigable)
        
        return grid
    
    def _compute_inflation(self):
        """
        Compute inflated occupancy grid using Minkowski sum.
        
        Returns:
            numpy array: Inflated grid (0=free, 1=obstacle)
        """
        # Create shapely polygons for obstacles
        # Only treat walls (1) and shelves (2) as obstacles - aisles (3), dock (4), staging (5) are free
        obstacles = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell_type = self.occupancy_grid[y, x]
                # Only walls and shelves are obstacles
                if cell_type == 1 or cell_type == 2:  # Wall or shelf
                    # Convert grid cell to world coordinates
                    world_x = x * self.grid_resolution
                    world_y = y * self.grid_resolution
                    # Create rectangle for cell
                    cell = box(
                        world_x,
                        world_y,
                        world_x + self.grid_resolution,
                        world_y + self.grid_resolution
                    )
                    obstacles.append(cell)
        
        if not obstacles:
            return self.occupancy_grid.copy()
        
        # Union all obstacles
        obstacle_union = unary_union(obstacles)
        
        # Create robot footprint for inflation
        robot_footprint = box(
            -self.robot_length / 2 - self.inflation_radius,
            -self.robot_width / 2 - self.inflation_radius,
            self.robot_length / 2 + self.inflation_radius,
            self.robot_width / 2 + self.inflation_radius
        )
        
        # Inflate obstacles using Minkowski sum (buffering)
        inflated_polygons = obstacle_union.buffer(self.inflation_radius)
        
        # Rasterize back to grid
        inflated_grid = np.zeros_like(self.occupancy_grid)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                world_x = (x + 0.5) * self.grid_resolution
                world_y = (y + 0.5) * self.grid_resolution
                point = Point(world_x, world_y)
                if inflated_polygons.contains(point) or inflated_polygons.distance(point) < 0.01:
                    inflated_grid[y, x] = 1
        
        return inflated_grid
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates."""
        gx = int(x / self.grid_resolution)
        gy = int(y / self.grid_resolution)
        gx = np.clip(gx, 0, self.grid_width - 1)
        gy = np.clip(gy, 0, self.grid_height - 1)
        return gx, gy
    
    def grid_to_world(self, gx, gy):
        """Convert grid coordinates to world coordinates."""
        x = (gx + 0.5) * self.grid_resolution
        y = (gy + 0.5) * self.grid_resolution
        return x, y
    
    def is_free(self, x, y, use_inflated=True):
        """Check if world position is free."""
        gx, gy = self.world_to_grid(x, y)
        grid = self.inflated_grid if use_inflated else self.occupancy_grid
        return grid[gy, gx] == 0
    
    def get_inflated_grid(self):
        """Get inflated occupancy grid for planning."""
        return self.inflated_grid.copy()
    
    def get_occupancy_grid(self):
        """Get original occupancy grid."""
        return self.occupancy_grid.copy()
    
    def compute_shortest_path_baseline(self, start, goal):
        """
        Compute shortest path baseline using Dijkstra on free grid.
        
        Args:
            start: (x, y) world coordinates
            goal: (x, y) world coordinates
            
        Returns:
            float: Shortest path length (for efficiency metric)
        """
        # Simple Dijkstra on 8-connected grid
        from scipy.sparse.csgraph import dijkstra
        from scipy.sparse import csr_matrix
        
        # Convert to grid coordinates
        sx, sy = self.world_to_grid(start[0], start[1])
        gx, gy = self.world_to_grid(goal[0], goal[1])
        
        # Create graph from free grid (8-connected)
        nodes = []
        node_to_grid = {}
        grid_to_node = np.full((self.grid_height, self.grid_width), -1, dtype=int)
        
        node_id = 0
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.occupancy_grid[y, x] == 0:  # free space
                    nodes.append((x, y))
                    grid_to_node[y, x] = node_id
                    node_to_grid[node_id] = (x, y)
                    node_id += 1
        
        n_nodes = len(nodes)
        if n_nodes == 0:
            return float('inf')
        
        # Build adjacency matrix (8-connected)
        edges = []
        weights = []
        for node_id, (x, y) in enumerate(nodes):
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        if self.occupancy_grid[ny, nx] == 0:
                            neighbor_id = grid_to_node[ny, nx]
                            if neighbor_id >= 0:
                                # Distance: 1 for horizontal/vertical, sqrt(2) for diagonal
                                dist = np.sqrt(dx*dx + dy*dy) * self.grid_resolution
                                edges.append((node_id, neighbor_id, dist))
        
        if not edges:
            return float('inf')
        
        # Build sparse matrix
        row_indices = [e[0] for e in edges]
        col_indices = [e[1] for e in edges]
        data = [e[2] for e in edges]
        graph = csr_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes))
        
        # Find nodes closest to start and goal
        start_node = grid_to_node[sy, sx]
        goal_node = grid_to_node[gy, gx]
        
        if start_node < 0 or goal_node < 0:
            return float('inf')
        
        # Run Dijkstra
        dist_matrix = dijkstra(graph, directed=False, indices=start_node)
        shortest_dist = dist_matrix[goal_node]
        
        self.shortest_path_baseline = shortest_dist if np.isfinite(shortest_dist) else float('inf')
        return self.shortest_path_baseline

