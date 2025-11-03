"""
Dynamic actors (humans, forklifts) for simulation.

Purpose: Spawn and manage dynamic obstacles with movement patterns,
         deterministic behavior via seeded RNG.

Inputs:
    - Actor config from YAML (count, speed ranges, behavior)
    - World boundaries
    - RNG seed for deterministic runs

Outputs:
    - Actor positions and velocities per frame
    - Polygon shapes for collision detection

Params:
    actor_config: dict - Actor configuration from YAML
    world_width: float - World width
    world_height: float - World height
    seed: int - RNG seed for deterministic behavior
"""

import numpy as np
from shapely.geometry import Polygon, Point, box
from typing import List, Tuple, Dict


class Actor:
    """Single dynamic actor (human or forklift)."""
    
    def __init__(self, x, y, width, length, speed, behavior_type, actor_type):
        """
        Initialize actor.
        
        Args:
            x, y: Initial position
            width, length: Actor dimensions
            speed: Movement speed
            behavior_type: 'crossing', 'random_walk', 'lane_follow', 'block_aisle'
            actor_type: 'human' or 'forklift'
        """
        self.x = x
        self.y = y
        self.width = width
        self.length = length
        self.speed = speed
        self.behavior_type = behavior_type
        self.actor_type = actor_type
        
        # Velocity
        self.vx = 0.0
        self.vy = 0.0
        
        # Behavior state
        self.target_x = x
        self.target_y = y
        self.path_progress = 0.0
        
    def get_footprint(self):
        """Get actor footprint as shapely polygon."""
        # Rectangle centered at (x, y)
        return box(
            self.x - self.length / 2,
            self.y - self.width / 2,
            self.x + self.length / 2,
            self.y + self.width / 2
        )
    
    def update(self, dt, world_width, world_height, obstacles):
        """
        Update actor position based on behavior.
        
        Args:
            dt: Time step
            world_width, world_height: World boundaries
            obstacles: List of obstacle polygons
        """
        if self.behavior_type == 'random_walk':
            self._update_random_walk(dt, world_width, world_height, obstacles)
        elif self.behavior_type == 'lane_follow':
            self._update_lane_follow(dt, world_width, world_height, obstacles)
        elif self.behavior_type == 'crossing':
            self._update_crossing(dt, world_width, world_height, obstacles)
        elif self.behavior_type == 'block_aisle':
            self._update_block_aisle(dt, world_width, world_height, obstacles)
        else:
            # Default: stationary
            pass
    
    def _update_random_walk(self, dt, world_width, world_height, obstacles):
        """Random walk behavior."""
        # Random direction changes occasionally
        if np.random.random() < 0.05:  # 5% chance to change direction
            angle = np.random.uniform(0, 2 * np.pi)
            self.vx = self.speed * np.cos(angle)
            self.vy = self.speed * np.sin(angle)
        
        # Update position
        new_x = self.x + self.vx * dt
        new_y = self.y + self.vy * dt
        
        # Bounce off walls
        if new_x < self.length/2 or new_x > world_width - self.length/2:
            self.vx = -self.vx
            new_x = np.clip(new_x, self.length/2, world_width - self.length/2)
        if new_y < self.width/2 or new_y > world_height - self.width/2:
            self.vy = -self.vy
            new_y = np.clip(new_y, self.width/2, world_height - self.width/2)
        
        self.x = new_x
        self.y = new_y
    
    def _update_lane_follow(self, dt, world_width, world_height, obstacles):
        """Lane-following behavior."""
        # Simple horizontal or vertical lane following
        if abs(self.vx) < 0.1:  # Start moving horizontally
            self.vx = self.speed * (1 if np.random.random() > 0.5 else -1)
            self.vy = 0.0
        
        new_x = self.x + self.vx * dt
        new_y = self.y + self.vy * dt
        
        # Bounce off walls
        if new_x < self.length/2 or new_x > world_width - self.length/2:
            self.vx = -self.vx
            new_x = np.clip(new_x, self.length/2, world_width - self.length/2)
        
        self.x = new_x
        self.y = new_y
    
    def _update_crossing(self, dt, world_width, world_height, obstacles):
        """Crossing behavior - moves across robot's likely path."""
        # Cross horizontally at mid-height
        target_y = world_height / 2
        if abs(self.y - target_y) > 0.1:
            # Move towards target y
            self.vy = self.speed * (1.0 if target_y > self.y else -1.0)
            self.vx = 0.0
        else:
            # Move horizontally
            self.vx = self.speed * (1 if np.random.random() > 0.5 else -1)
            self.vy = 0.0
        
        new_x = self.x + self.vx * dt
        new_y = self.y + self.vy * dt
        
        # Clamp to world bounds
        new_x = np.clip(new_x, self.length/2, world_width - self.length/2)
        new_y = np.clip(new_y, self.width/2, world_height - self.width/2)
        
        self.x = new_x
        self.y = new_y
    
    def _update_block_aisle(self, dt, world_width, world_height, obstacles):
        """Block aisle behavior - stays in a blocking position."""
        # Move to a position that blocks an aisle, then stop
        target_x = world_width * 0.4  # Block first aisle
        target_y = world_height * 0.5
        
        dx = target_x - self.x
        dy = target_y - self.y
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist > 0.2:
            # Move towards blocking position
            self.vx = self.speed * dx / dist
            self.vy = self.speed * dy / dist
            new_x = self.x + self.vx * dt
            new_y = self.y + self.vy * dt
            self.x = new_x
            self.y = new_y
        else:
            # Stay in blocking position
            self.vx = 0.0
            self.vy = 0.0


class ActorManager:
    """Manager for all dynamic actors."""
    
    def __init__(self, world_width, world_height, actor_config, seed=42):
        """
        Initialize actor manager.
        
        Args:
            world_width, world_height: World dimensions
            actor_config: Dictionary from config/actors.yaml
            seed: RNG seed for deterministic behavior
        """
        self.world_width = world_width
        self.world_height = world_height
        np.random.seed(seed)
        
        self.actors: List[Actor] = []
        
        # Spawn actors based on config
        if 'actors' in actor_config and actor_config['actors']:
            for actor_group in actor_config['actors']:
                actor_type = actor_group.get('type', 'human')
                count = actor_group.get('count', 1)
                speed_range = actor_group.get('speed_range', [0.5, 0.8])
                behavior = actor_group.get('behavior', 'random_walk')
                spawn_pos = actor_group.get('spawn_pos', None)
                
                # Get actor dimensions
                actor_types = actor_config.get('actor_types', {})
                if actor_type in actor_types:
                    width = actor_types[actor_type].get('width', 0.4)
                    length = actor_types[actor_type].get('length', 0.4)
                else:
                    width = 0.4
                    length = 0.4
                
                # Spawn actors
                for i in range(count):
                    if spawn_pos:
                        x, y = spawn_pos[0], spawn_pos[1]
                    else:
                        # Random spawn position (avoid walls)
                        margin = 2.0
                        x = np.random.uniform(margin, world_width - margin)
                        y = np.random.uniform(margin, world_height - margin)
                    
                    speed = np.random.uniform(speed_range[0], speed_range[1])
                    
                    actor = Actor(x, y, width, length, speed, behavior, actor_type)
                    self.actors.append(actor)
    
    def update(self, dt):
        """Update all actors."""
        obstacles = []  # Could add static obstacles here
        for actor in self.actors:
            actor.update(dt, self.world_width, self.world_height, obstacles)
    
    def get_actor_polygons(self):
        """Get all actor footprints as polygons for collision detection."""
        return [actor.get_footprint() for actor in self.actors]
    
    def get_actors(self):
        """Get list of all actors."""
        return self.actors

