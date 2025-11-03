"""
LiDAR sensor for robot perception.

Purpose: 2D LiDAR raycast, min-clearance monitoring, re-planning triggers.

Inputs:
    - Robot position (x, y, theta)
    - World obstacles
    - Dynamic actor polygons

Outputs:
    - Range measurements per beam
    - Minimum clearance to obstacles

Params:
    fov: float - Field of view in degrees (360 = full circle)
    resolution: float - Degrees per beam
    max_range: float - Maximum range in meters/pixels
    update_rate: float - Update rate in Hz
"""

import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union


class LiDARSensor:
    """2D LiDAR sensor with raycast."""
    
    def __init__(self, fov=360.0, resolution=1.0, max_range=5.0, update_rate=20.0):
        """
        Initialize LiDAR sensor.
        
        Args:
            fov: Field of view in degrees (360 = full circle)
            resolution: Degrees per beam
            max_range: Maximum range in meters/pixels
            update_rate: Update rate in Hz
        """
        self.fov = fov
        self.resolution = resolution
        self.max_range = max_range
        self.update_rate = update_rate
        self.dt_accumulator = 0.0
        self.dt_threshold = 1.0 / update_rate
        
        # Compute number of beams
        self.num_beams = int(fov / resolution)
        
        # Angles for each beam
        start_angle = -fov / 2.0
        self.beam_angles = np.deg2rad(
            start_angle + np.arange(self.num_beams) * resolution
        )
    
    def scan(self, robot_x, robot_y, robot_theta, world, actor_polygons):
        """
        Perform LiDAR scan - optimized version.
        
        Args:
            robot_x, robot_y: Robot position
            robot_theta: Robot orientation (radians)
            world: World object
            actor_polygons: List of actor polygons for collision
        
        Returns:
            (ranges, min_clearance): Array of ranges and minimum clearance
        """
        ranges = []
        
        # Use cached static obstacle union if available (only rebuild when world changes)
        if not hasattr(self, '_cached_obstacle_union') or not hasattr(self, '_last_world_hash'):
            # Build static obstacle union (do this once or rarely)
            static_obstacles = []
            for y in range(world.grid_height):
                for x in range(world.grid_width):
                    if world.inflated_grid[y, x] == 1:
                        wx, wy = world.grid_to_world(x, y)
                        cell_size = world.grid_resolution
                        from shapely.geometry import box
                        cell = box(
                            wx - cell_size/2,
                            wy - cell_size/2,
                            wx + cell_size/2,
                            wy + cell_size/2
                        )
                        static_obstacles.append(cell)
            
            if static_obstacles:
                self._cached_obstacle_union = unary_union(static_obstacles)
            else:
                self._cached_obstacle_union = None
            
            # Simple hash of world state (just use grid sum as hash)
            self._last_world_hash = world.inflated_grid.sum()
        
        # Combine with dynamic actors (only actors change frequently)
        all_obstacles = []
        if self._cached_obstacle_union is not None:
            # Add cached static obstacles
            if hasattr(self._cached_obstacle_union, 'geoms'):
                all_obstacles.extend(self._cached_obstacle_union.geoms)
            else:
                all_obstacles.append(self._cached_obstacle_union)
        
        # Add dynamic actors
        all_obstacles.extend(actor_polygons)
        
        # Union all obstacles for faster intersection (only if we have obstacles)
        if all_obstacles:
            # Only union if we have multiple obstacles (optimization)
            if len(all_obstacles) > 1:
                obstacle_union = unary_union(all_obstacles)
            else:
                obstacle_union = all_obstacles[0]
        else:
            obstacle_union = None
        
        # Cast rays
        min_range = float('inf')
        
        for angle in self.beam_angles:
            # Ray angle in world frame
            ray_angle = robot_theta + angle
            
            # Ray endpoint
            end_x = robot_x + self.max_range * np.cos(ray_angle)
            end_y = robot_y + self.max_range * np.sin(ray_angle)
            
            # Create ray
            ray = LineString([(robot_x, robot_y), (end_x, end_y)])
            
            # Intersect with obstacles
            range_measurement = self.max_range
            
            if obstacle_union is not None:
                # Find intersection
                intersection = obstacle_union.intersection(ray)
                if not intersection.is_empty:
                    # Find closest intersection point
                    if hasattr(intersection, 'geoms'):
                        # Multiple intersections
                        closest_dist = float('inf')
                        for geom in intersection.geoms:
                            if hasattr(geom, 'coords'):
                                for coord in geom.coords:
                                    dist = np.sqrt(
                                        (coord[0] - robot_x)**2 + (coord[1] - robot_y)**2
                                    )
                                    closest_dist = min(closest_dist, dist)
                        range_measurement = closest_dist
                    else:
                        # Single intersection
                        if hasattr(intersection, 'coords'):
                            coords = list(intersection.coords)
                            if coords:
                                closest_point = coords[0]
                                range_measurement = np.sqrt(
                                    (closest_point[0] - robot_x)**2 +
                                    (closest_point[1] - robot_y)**2
                                )
            
            ranges.append(range_measurement)
            min_range = min(min_range, range_measurement)
        
        # Compute minimum clearance (minimum distance to any obstacle)
        min_clearance = min_range if min_range < float('inf') else self.max_range
        
        return np.array(ranges), min_clearance
    
    def should_update(self, dt):
        """Check if sensor should update based on update rate."""
        self.dt_accumulator += dt
        if self.dt_accumulator >= self.dt_threshold:
            self.dt_accumulator = 0.0
            return True
        return False

