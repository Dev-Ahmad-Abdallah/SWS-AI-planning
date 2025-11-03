"""
Pygame simulation engine.

Purpose: Main simulation loop, rendering, input handling, component integration.

Inputs:
    - Scenario ID (S1-S4)
    - Planner type (a_star, prm)
    - Config preset (baseline, safe, fast)

Outputs:
    - Visual simulation display
    - Logged KPIs to CSV
    - Optional video capture

Params:
    scenario: str - Scenario identifier
    planner: str - Global planner type
    config_preset: str - Configuration preset name
"""

import pygame
import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from sim.world import World
from sim.actors import ActorManager
from robot.kinematics import RobotKinematics
from robot.sensors import LiDARSensor
from nav.task_fsm import TaskFSM
from nav.a_star import AStarPlanner
from nav.prm import PRMPlanner
from nav.bfs import BFSPlanner
from nav.dfs import DFSPlanner
from nav.dijkstra import DijkstraPlanner
from nav.rrt import RRTPlanner
from nav.spanning_tree import (
    MSTPlanner, DFSSpanningTreePlanner, BFSSpanningTreePlanner,
    MaximumSpanningTreePlanner, RootedSpanningTreePlanner
)
from nav.slam import SLAM
from nav.local_dwa import LocalDWA
from nav.maps import MapUtils
from tools.logger import KPILogger
from tools.metrics import MetricsTracker


class SimulationEngine:
    """Main Pygame simulation engine."""
    
    def __init__(self, scenario: str, planner: str, config_preset: str = "baseline"):
        """
        Initialize simulation engine.
        
        Args:
            scenario: Scenario ID (S1, S2, S3, S4)
            planner: Planner type ('a_star' or 'prm')
            config_preset: Config preset ('baseline', 'safe', 'fast')
        """
        self.scenario = scenario
        self.planner_type = planner
        self.config_preset = config_preset
        
        # Load configuration
        config_path = Path(f"config/{config_preset}.yaml")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load actor configuration
        actors_path = Path("config/actors.yaml")
        with open(actors_path, 'r') as f:
            actors_config_all = yaml.safe_load(f)
            self.actor_config = actors_config_all.get(scenario, {})
        
        # Initialize Pygame
        pygame.init()
        self.width = self.config['sim']['window_width']
        self.height = self.config['sim']['window_height']
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Smart Warehouse Robot - {scenario} - {planner}")
        self.clock = pygame.time.Clock()
        
        # Simulation state
        self.running = True
        self.paused = False
        self.recording = False
        self.frame_count = 0
        self.sim_time = 0.0
        self.dt = self.config['sim']['dt']
        
        # Initialize world
        self.world = World(
            width=40.0,
            height=30.0,
            grid_resolution=self.config['sim']['grid_resolution'],
            robot_width=self.config['robot']['width'],
            robot_length=self.config['robot']['length'],
            inflation_radius=self.config['inflation']['radius']
        )
        
        # Initialize actors
        seed = self.actor_config.get('seed', self.config.get('seed', 42))
        self.actor_manager = ActorManager(
            world_width=self.world.width,
            world_height=self.world.height,
            actor_config=self.actor_config,
            seed=seed
        )
        
        # Initialize robot at DOCK AREA (bottom center, free space)
        # Dock is at bottom center: x = 50% width, y = 90% height
        world_width = self.world.width
        world_height = self.world.height
        robot_start_x = world_width * 0.5  # Center of dock
        robot_start_y = world_height * 0.92  # Near bottom in dock area
        
        # Ensure starting position is free
        if not self.world.is_free(robot_start_x, robot_start_y):
            # Try nearby positions if dock is occupied
            for offset_x in [0, -1, 1, -2, 2]:
                for offset_y in [0, -0.5, 0.5, -1, 1]:
                    test_x = robot_start_x + offset_x
                    test_y = robot_start_y + offset_y
                    if 0 < test_x < world_width and 0 < test_y < world_height:
                        if self.world.is_free(test_x, test_y):
                            robot_start_x = test_x
                            robot_start_y = test_y
                            break
                else:
                    continue
                break
        
        self.robot = RobotKinematics(
            x=robot_start_x,
            y=robot_start_y,
            theta=0.0,  # Facing north initially
            v_max=self.config['robot']['v_max'],
            omega_max=self.config['robot']['omega_max'],
            width=self.config['robot']['width'],
            length=self.config['robot']['length']
        )
        
        # Initialize sensors (optimized - reduce beams for performance)
        lidar_resolution = max(2.0, self.config['lidar']['resolution'])  # Reduce beams
        self.lidar = LiDARSensor(
            fov=self.config['lidar']['fov'],
            resolution=lidar_resolution,  # Fewer beams for better performance
            max_range=self.config['lidar']['max_range'],
            update_rate=self.config['lidar']['update_rate']
        )
        
        # Initialize planners
        self.map_utils = MapUtils(self.world)
        
        # Initialize SLAM for mapping (if enabled)
        self.use_slam = False  # Can be enabled for exploration mode
        self.slam = None
        if self.use_slam:
            self.slam = SLAM(
                world_width=self.world.width,
                world_height=self.world.height,
                grid_resolution=self.world.grid_resolution,
                max_range=self.config['lidar']['max_range']
            )
        
        # Initialize global planner based on selection
        if planner == 'bfs':
            self.global_planner = BFSPlanner(self.world.inflated_grid, self.world.grid_resolution)
        elif planner == 'dfs':
            self.global_planner = DFSPlanner(self.world.inflated_grid, self.world.grid_resolution)
        elif planner == 'dijkstra':
            self.global_planner = DijkstraPlanner(self.world.inflated_grid, self.world.grid_resolution)
        elif planner == 'a_star':
            self.global_planner = AStarPlanner(self.world.inflated_grid, self.world.grid_resolution)
        elif planner == 'rrt':
            # RRT uses inflated grid but checks for obstacles correctly
            self.global_planner = RRTPlanner(
                world_width=self.world.width,
                world_height=self.world.height,
                occupancy_grid=self.world.inflated_grid,  # Use inflated grid for collision checking
                grid_resolution=self.world.grid_resolution,
                max_iterations=10000,  # More iterations for complex environments
                step_size=0.8,  # Larger step size for faster planning
                goal_bias=0.2  # More goal bias to find path faster
            )
        elif planner == 'prm':
            self.global_planner = PRMPlanner(
                world_width=self.world.width,
                world_height=self.world.height,
                occupancy_grid=self.world.inflated_grid,
                grid_resolution=self.world.grid_resolution,
                num_samples=500,
                k_neighbors=10
            )
        elif planner == 'mst':
            self.global_planner = MSTPlanner(self.world.inflated_grid, self.world.grid_resolution)
        elif planner == 'dfs_spanning':
            self.global_planner = DFSSpanningTreePlanner(self.world.inflated_grid, self.world.grid_resolution)
        elif planner == 'bfs_spanning':
            self.global_planner = BFSSpanningTreePlanner(self.world.inflated_grid, self.world.grid_resolution)
        elif planner == 'max_spanning':
            self.global_planner = MaximumSpanningTreePlanner(self.world.inflated_grid, self.world.grid_resolution)
        elif planner == 'rooted_spanning':
            self.global_planner = RootedSpanningTreePlanner(self.world.inflated_grid, self.world.grid_resolution)
        else:
            raise ValueError(f"Unknown planner: {planner}")
        
        # Initialize local avoidance
        self.local_avoider = LocalDWA(
            v_max=self.config['robot']['v_max'],
            omega_max=self.config['robot']['omega_max'],
            w_goal=self.config['local_avoid']['w_goal'],
            w_obs=self.config['local_avoid']['w_obs'],
            w_smooth=self.config['local_avoid']['w_smooth'],
            lookahead=self.config['local_avoid']['lookahead'],
            min_clearance=self.config['local_avoid']['min_clearance'],
            world=self.world  # Pass world object for collision checking
        )
        
        # Initialize task FSM with realistic warehouse bay positions
        # Bays in different storage zones (in aisles between shelves)
        world_width = self.world.width
        world_height = self.world.height
        
        # REALISTIC BAY POSITIONS: In free aisles/corridors (verify they're free!)
        # Place bays in the main vertical corridor and horizontal cross-corridors
        bay_positions = []
        
        # Bay 1: Main corridor, middle height
        bay1_x, bay1_y = world_width * 0.5, world_height * 0.35
        if self.world.is_free(bay1_x, bay1_y):
            bay_positions.append((bay1_x, bay1_y))
        
        # Bay 2: Main corridor, upper height
        bay2_x, bay2_y = world_width * 0.5, world_height * 0.25
        if self.world.is_free(bay2_x, bay2_y):
            bay_positions.append((bay2_x, bay2_y))
        
        # Bay 3: Left side aisle
        bay3_x, bay3_y = world_width * 0.25, world_height * 0.35
        if self.world.is_free(bay3_x, bay3_y):
            bay_positions.append((bay3_x, bay3_y))
        
        # Bay 4: Right side aisle
        bay4_x, bay4_y = world_width * 0.75, world_height * 0.35
        if self.world.is_free(bay4_x, bay4_y):
            bay_positions.append((bay4_x, bay4_y))
        
        # Bay 5: Top cross-corridor
        bay5_x, bay5_y = world_width * 0.5, world_height * 0.15
        if self.world.is_free(bay5_x, bay5_y):
            bay_positions.append((bay5_x, bay5_y))
        
        # If no valid bays found, use fallback positions in main corridor
        if not bay_positions:
            print("WARNING: No valid bay positions found! Using fallback positions.")
            bay_positions = [
                (world_width * 0.5, world_height * 0.30),
                (world_width * 0.5, world_height * 0.20),
                (world_width * 0.5, world_height * 0.40),
            ]
        
        # Create realistic missions (pickup-delivery pairs)
        self.task_fsm = TaskFSM(
            bays=bay_positions,
            dock=(world_width * 0.5, world_height * 0.92),  # Dock at bottom center
            mission_type="delivery"  # Use realistic pickup-delivery missions
        )
        
        # Initialize logging and metrics
        self.metrics_tracker = MetricsTracker(
            shortest_path_baseline=None  # Will be computed after first planning
        )
        self.logger = KPILogger(scenario=scenario, planner=planner)
        
        # Navigation state
        self.current_path = None
        self.current_waypoint_idx = 0
        self.needs_replan = False
        self.last_plan_time = 0.0
        self.replan_interval = 2.0  # Minimum time between re-plans (increased to prevent excessive replans)
        self._last_goal = None  # Track last goal to prevent unnecessary replanning
        self._last_plan_dist_to_goal = None  # Track distance to goal at last plan
        
        # Cache for sensors
        self.last_lidar_ranges = []
        self.last_min_clearance = float('inf')
        
        # Cache for local avoidance
        self.last_v = 0.0
        self.last_omega = 0.0
        
        # Cache for actor polygons (updated periodically)
        self._cached_actor_polygons = []
        
        # Cache LiDAR beam angles for rendering
        self.beam_angles = None
        
        # Camera - larger scale for better visibility of 2D grid
        self.camera_x = robot_start_x
        self.camera_y = robot_start_y
        self.camera_scale = 40.0  # pixels per meter - larger for clearer grid cells
        
    def run(self):
        """Main simulation loop - GUARANTEED 60 FPS with real-time speed."""
        target_fps = 60
        
        while self.running:
            # Get actual frame delta for real-time simulation - FORCE 60 FPS
            dt_actual = self.clock.tick(target_fps) / 1000.0  # Convert ms to seconds
            
            # Use actual dt for REAL-TIME physics (not fixed dt)
            if not self.paused:
                # Update simulation with REAL-TIME dt for accurate movement
                self._update(dt_actual)  # Use actual elapsed time for real-time speed
                self.frame_count += 1
                # Update sim_time with actual elapsed time for accurate timing
                self.sim_time += dt_actual
                
                # Update metrics every frame for accurate tracking
                self._update_metrics()
            
            # Handle input
            self._handle_input()
            
            # Render
            self._render()
            
            # Check for completion
            if self.task_fsm.is_done():
                print(f"Mission completed in {self.sim_time:.2f} seconds")
                break
        
        # Cleanup
        self._cleanup()
    
    def _handle_input(self):
        """Handle keyboard and mouse input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Restart simulation
                    self._restart()
                elif event.key == pygame.K_v:
                    # Toggle video recording
                    self.recording = not self.recording
                    print(f"Recording: {'ON' if self.recording else 'OFF'}")
                elif event.key == pygame.K_SPACE:
                    # Pause/unpause
                    self.paused = not self.paused
    
    def _update(self, dt):
        """Update simulation step."""
        # Update actors
        self.actor_manager.update(dt)
        
        # Update actor polygons every frame for accurate perception
        self._cached_actor_polygons = self.actor_manager.get_actor_polygons()
        
        # Update robot sensors (only at update rate, not every frame)
        if self.lidar.should_update(dt):
            lidar_ranges, min_clearance = self.lidar.scan(
                self.robot.x, self.robot.y, self.robot.theta,
                self.world, self._cached_actor_polygons
            )
            self.last_lidar_ranges = lidar_ranges
            self.last_min_clearance = min_clearance
            # Cache beam angles for rendering
            if hasattr(self.lidar, 'beam_angles'):
                self.beam_angles = self.lidar.beam_angles
            
            # Update SLAM if enabled
            if self.slam is not None and hasattr(self.lidar, 'beam_angles'):
                self.slam.update(
                    self.robot.x, self.robot.y, self.robot.theta,
                    lidar_ranges, self.lidar.beam_angles
                )
        else:
            lidar_ranges = self.last_lidar_ranges if hasattr(self, 'last_lidar_ranges') else []
            min_clearance = self.last_min_clearance if hasattr(self, 'last_min_clearance') else float('inf')
        
        # OPTIMIZED: More aggressive replanning triggers to prevent getting stuck
        # Check multiple conditions for replanning
        clearance_low = min_clearance < self.config['replan']['threshold_clearance']
        clearance_dangerous = min_clearance < self.config['local_avoid']['min_clearance'] * 0.6
        time_since_last_plan = self.sim_time - self.last_plan_time
        
        # Check if DWA thinks we should replan
        dwa_should_replan = False
        if self.current_path:
            progress = self._compute_progress()
            dwa_should_replan = self.local_avoider.should_replan(progress, min_clearance)
        
        # Trigger replan if:
        # 1. Clearance is dangerously low (immediate replan)
        # 2. Clearance is low AND we haven't replanned recently AND DWA suggests replan
        # 3. DWA thinks we should replan AND enough time has passed
        if clearance_dangerous:
            if time_since_last_plan > self.replan_interval * 0.5:  # Allow faster replan when dangerous
                self.needs_replan = True
        elif clearance_low and dwa_should_replan:
            if time_since_last_plan > self.replan_interval:
                self.needs_replan = True
        elif dwa_should_replan and time_since_last_plan > self.replan_interval:
            self.needs_replan = True
        else:
            # Clear replan flag if clearance is good and we're making progress
            if min_clearance > self.config['replan']['threshold_clearance'] * 1.5:
                self.needs_replan = False
        
        # Update task FSM (full update every frame for realism)
        prev_state = self.task_fsm.state
        self.task_fsm.update(self.robot.x, self.robot.y, self.robot.theta, self.sim_time)
        
        # CRITICAL FIX: If state changed (e.g., INSPECT -> GOTO_BAY), clear path and force replan
        # BUT: Only if the new goal is different from the current goal
        if prev_state != self.task_fsm.state:
            new_goal = self.task_fsm.get_current_goal()
            old_goal = self._last_goal
            
            # Only replan if goal actually changed or we're transitioning to a new task
            from nav.task_fsm import TaskState
            goal_changed = (old_goal is None or 
                          (new_goal is not None and old_goal is not None and 
                           (abs(new_goal[0] - old_goal[0]) > 0.5 or abs(new_goal[1] - old_goal[1]) > 0.5)) or
                          prev_state in [TaskState.LOAD, TaskState.UNLOAD, TaskState.INSPECT])
            
            if goal_changed:
                print(f"State changed: {prev_state.name if hasattr(prev_state, 'name') else prev_state} -> {self.task_fsm.state.name if hasattr(self.task_fsm.state, 'name') else self.task_fsm.state}")
                self.current_path = None  # Clear old path
                self.current_waypoint_idx = 0  # Reset waypoint index
                self.needs_replan = True  # Force immediate replan
                # Store current goal for next comparison
                self._last_goal = new_goal
            else:
                # State changed but goal is the same (e.g., reached waypoint in path) - don't replan
                self.needs_replan = False
        
        # Check if we need to plan - CRITICAL FIX: Prevent excessive replanning
        goal = self.task_fsm.get_current_goal()
        at_goal = False
        moving_towards_goal = False
        
        if goal is not None:
            dist_to_goal = np.sqrt(
                (self.robot.x - goal[0])**2 + (self.robot.y - goal[1])**2
            )
            at_goal = dist_to_goal < 0.8  # Close enough to goal (increased threshold)
            
            # Check if we're making progress towards goal (prevent replan when moving)
            if self.current_path is not None and len(self.current_path) > 0:
                # If we have a path and it leads to goal, check if we're following it
                last_waypoint = self.current_path[-1]
                waypoint_dist_to_goal = np.sqrt(
                    (last_waypoint[0] - goal[0])**2 + (last_waypoint[1] - goal[1])**2
                )
                # If path ends near goal and we're moving, don't replan
                if waypoint_dist_to_goal < 1.0:
                    # Check if we moved closer to goal since last plan
                    if hasattr(self, '_last_plan_dist_to_goal'):
                        if dist_to_goal < self._last_plan_dist_to_goal * 1.1:  # Getting closer or same
                            moving_towards_goal = True
                    else:
                        self._last_plan_dist_to_goal = dist_to_goal
                        moving_towards_goal = True
        
        # IMPORTANT: Plan immediately on first frame if no path exists or state changed
        # BUT: Don't replan if we're already at the goal or making progress towards it
        should_replan = (self.current_path is None or self.needs_replan) and not at_goal and not moving_towards_goal
        
        if should_replan:
            if self.sim_time - self.last_plan_time > self.replan_interval or self.current_path is None:
                self._plan_path()
                self.needs_replan = False
                self.last_plan_time = self.sim_time
                if goal is not None:
                    self._last_plan_dist_to_goal = np.sqrt(
                        (self.robot.x - goal[0])**2 + (self.robot.y - goal[1])**2
                    )
        
        # Execute local avoidance (optimized for 60 FPS - reduced samples)
        # CRITICAL FIX: Force immediate backward movement if clearance is dangerously low
        emergency_clearance = 0.3  # Emergency threshold - immediate backward
        if min_clearance < emergency_clearance:
            # EMERGENCY: Immediate fast backward movement
            v = -self.config['robot']['v_max'] * 0.8  # Fast backward movement (increased from 0.3)
            omega = self.config['robot']['omega_max'] * 0.3  # Turn slightly while backing up
            self.last_v = v
            self.last_omega = omega
        # CRITICAL FIX: If in WAIT state (deadlock recovery), force backward movement
        elif self.task_fsm.state.name == 'WAIT' if hasattr(self.task_fsm.state, 'name') else False:
            # Force backward movement to get unstuck during deadlock recovery
            v = -self.config['robot']['v_max'] * 0.6  # Faster backward movement (increased from 0.3)
            omega = 0.0  # Straight back
            self.last_v = v
            self.last_omega = omega
        elif self.current_path and len(self.current_path) > 0:
            next_waypoint = self.current_path[min(self.current_waypoint_idx, len(self.current_path) - 1)]
            # Pass actor polygons for dynamic obstacle avoidance
            v, omega = self.local_avoider.compute(
                self.robot.x, self.robot.y, self.robot.theta,
                next_waypoint[0], next_waypoint[1],
                lidar_ranges, min_clearance, self._cached_actor_polygons
            )
            
            # FALLBACK: If DWA returns zero, use simple direct movement towards waypoint
            if abs(v) < 0.01 and abs(omega) < 0.01:
                # Direct movement towards waypoint
                dx = next_waypoint[0] - self.robot.x
                dy = next_waypoint[1] - self.robot.y
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist > 0.1:  # Not at waypoint yet
                    # Compute desired heading
                    desired_theta = np.arctan2(dy, dx)
                    angle_diff = desired_theta - self.robot.theta
                    
                    # Normalize angle to [-pi, pi]
                    while angle_diff > np.pi:
                        angle_diff -= 2 * np.pi
                    while angle_diff < -np.pi:
                        angle_diff += 2 * np.pi
                    
                    # CRITICAL FIX: Only rotate if angle difference is significant AND we have clearance
                    # Don't rotate unnecessarily when sensors show no obstacles
                    # Only rotate if angle is large AND we're safe OR if we're close to waypoint
                    if abs(angle_diff) > 0.5:  # Only turn if angle is SIGNIFICANT (was 0.2)
                        # Check if we have clearance before turning in place
                        if min_clearance > self.config['local_avoid']['min_clearance']:
                            omega = np.clip(1.5 * angle_diff, -self.config['robot']['omega_max'], self.config['robot']['omega_max'])
                            v = 0.0  # Stop while turning
                        else:
                            # Not safe to turn in place - move forward while turning slightly
                            v = np.clip(0.3 * self.config['robot']['v_max'], 0.0, self.config['robot']['v_max'])  # Slow forward
                            omega = np.clip(0.5 * angle_diff, -self.config['robot']['omega_max'], self.config['robot']['omega_max'])  # Slow turn
                    elif abs(angle_diff) > 0.15:  # Medium angle - move forward with slight correction
                        v = np.clip(0.8 * self.config['robot']['v_max'], 0.0, self.config['robot']['v_max'])
                        omega = np.clip(0.3 * angle_diff, -self.config['robot']['omega_max'], self.config['robot']['omega_max'])  # Gentle correction
                    else:  # Small angle - just move forward straight
                        v = np.clip(0.95 * self.config['robot']['v_max'], 0.0, self.config['robot']['v_max'])
                        omega = 0.0  # NO rotation if angle is small - prevent unnecessary spinning!
            
            # Store velocity commands for display
            self.last_v = v
            self.last_omega = omega
        else:
            v, omega = 0.0, 0.0
            self.last_v = 0.0
            self.last_omega = 0.0
        
        # OPTIMIZED: Better progress monitoring for replanning
        if self.current_path:
            progress = self._compute_progress()
            # Check if progress is too low (blocked or stuck)
            progress_stuck = progress < self.config['replan']['threshold_progress']
            
            # Also check if we're not moving towards goal
            if progress_stuck and goal is not None:
                # Check if we're actually making progress towards goal
                current_dist = np.sqrt(
                    (self.robot.x - goal[0])**2 + (self.robot.y - goal[1])**2
                )
                time_since_last_plan = self.sim_time - self.last_plan_time
                if hasattr(self, '_last_progress_check_dist'):
                    # If distance to goal hasn't decreased significantly, we're stuck
                    dist_reduction = self._last_progress_check_dist - current_dist
                    if dist_reduction < 0.1 and time_since_last_plan > self.replan_interval * 0.7:
                        self.needs_replan = True
                    self._last_progress_check_dist = current_dist
                else:
                    self._last_progress_check_dist = current_dist
            
            # Original progress check
            time_since_last_plan = self.sim_time - self.last_plan_time
            if progress_stuck and time_since_last_plan > self.replan_interval:
                self.needs_replan = True
        
        # Update robot kinematics - FORCE MOVEMENT
        self.robot.update(v, omega, dt, self.world)
        
        # Update camera to follow robot
        self.camera_x = self.robot.x
        self.camera_y = self.robot.y
        
        # Check if waypoint reached (optimize - avoid sqrt if possible)
        if self.current_path and self.current_waypoint_idx < len(self.current_path):
            waypoint = self.current_path[self.current_waypoint_idx]
            dx = self.robot.x - waypoint[0]
            dy = self.robot.y - waypoint[1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < 0.25:  # 0.5^2 = 0.25 (avoid sqrt)
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx >= len(self.current_path):
                    self.current_path = None  # Path complete
    
    def _plan_path(self):
        """Plan global path using selected planner."""
        current_pos = (self.robot.x, self.robot.y)
        goal = self.task_fsm.get_current_goal()
        
        if goal is None:
            print(f"WARNING: No goal available! Task state: {self.task_fsm.state.name if hasattr(self.task_fsm.state, 'name') else self.task_fsm.state}")
            return
        
        print(f"Planning path from ({current_pos[0]:.2f}, {current_pos[1]:.2f}) to ({goal[0]:.2f}, {goal[1]:.2f})")
        
        # Verify start and goal are free
        if not self.world.is_free(current_pos[0], current_pos[1]):
            print(f"ERROR: Start position ({current_pos[0]:.2f}, {current_pos[1]:.2f}) is not free!")
            # Find nearest free position
            for offset in [(0, 0), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                test_pos = (current_pos[0] + offset[0], current_pos[1] + offset[1])
                if self.world.is_free(test_pos[0], test_pos[1]):
                    current_pos = test_pos
                    print(f"  Using nearby free position: ({test_pos[0]:.2f}, {test_pos[1]:.2f})")
                    break
        
        if not self.world.is_free(goal[0], goal[1]):
            print(f"ERROR: Goal position ({goal[0]:.2f}, {goal[1]:.2f}) is not free!")
            # Find nearest free position to goal
            for offset in [(0, 0), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                test_goal = (goal[0] + offset[0], goal[1] + offset[1])
                if self.world.is_free(test_goal[0], test_goal[1]):
                    goal = test_goal
                    print(f"  Using nearby free goal: ({test_goal[0]:.2f}, {test_goal[1]:.2f})")
                    break
        
        # Plan path
        import time
        start_time = time.time()
        
        # Determine if planner uses grid or world coordinates
        grid_based_planners = ['bfs', 'dfs', 'dijkstra', 'a_star', 'mst', 
                               'dfs_spanning', 'bfs_spanning', 'max_spanning', 'rooted_spanning']
        world_based_planners = ['rrt', 'prm']
        
        if self.planner_type in grid_based_planners:
            # Convert to grid coordinates for planning
            start_grid = self.world.world_to_grid(current_pos[0], current_pos[1])
            goal_grid = self.world.world_to_grid(goal[0], goal[1])
            
            # Verify grid positions are valid
            if (start_grid[0] < 0 or start_grid[0] >= self.world.grid_width or 
                start_grid[1] < 0 or start_grid[1] >= self.world.grid_height):
                print(f"ERROR: Invalid start grid position: {start_grid}")
                self.current_path = None
                return
            
            if (goal_grid[0] < 0 or goal_grid[0] >= self.world.grid_width or 
                goal_grid[1] < 0 or goal_grid[1] >= self.world.grid_height):
                print(f"ERROR: Invalid goal grid position: {goal_grid}")
                self.current_path = None
                return
            
            path_grid = self.global_planner.plan(
                tuple(start_grid), tuple(goal_grid)
            )
            # Convert grid path to world coordinates
            if path_grid:
                self.current_path = [
                    self.world.grid_to_world(gx, gy) for gx, gy in path_grid
                ]
            else:
                self.current_path = None
        else:  # world-based planners (RRT, PRM)
            path_grid = self.global_planner.plan(current_pos, goal)
            self.current_path = path_grid  # Already in world coords
        
        plan_time = (time.time() - start_time) * 1000  # milliseconds
        
        if self.current_path:
            self.current_waypoint_idx = 0
            self.metrics_tracker.record_plan_time(plan_time)
            self.metrics_tracker.record_replan()
            self._last_goal = goal  # Store the goal we just planned to
            print(f"Path planned: {len(self.current_path)} waypoints, time: {plan_time:.2f}ms")
        else:
            print(f"WARNING: Path planning FAILED! No path found from ({current_pos[0]:.2f}, {current_pos[1]:.2f}) to ({goal[0]:.2f}, {goal[1]:.2f})")
    
    def _compute_progress(self):
        """Compute progress along current path - optimized."""
        if not self.current_path:
            return 0.0
        
        # Distance to next waypoint (avoid sqrt if possible)
        if self.current_waypoint_idx < len(self.current_path):
            waypoint = self.current_path[self.current_waypoint_idx]
            dx = self.robot.x - waypoint[0]
            dy = self.robot.y - waypoint[1]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq)  # Only compute sqrt once
            return max(0.0, 1.0 - dist / 5.0)  # Normalized progress
        
        return 0.0
    
    def _update_metrics(self):
        """Update metrics tracker."""
        # Track path length
        if self.current_path:
            total_path_len = 0.0
            prev = (self.robot.x, self.robot.y)
            for waypoint in self.current_path[self.current_waypoint_idx:]:
                total_path_len += np.sqrt(
                    (waypoint[0] - prev[0])**2 + (waypoint[1] - prev[1])**2
                )
                prev = waypoint
            self.metrics_tracker.update_path_length(total_path_len)
        
        # Track min clearance (use cached value)
        min_clear = self.last_min_clearance if hasattr(self, 'last_min_clearance') else float('inf')
        self.metrics_tracker.update_min_clearance(min_clear)
        
        # Check for collisions every frame for safety
        robot_footprint = self.robot.get_footprint()
        actor_polygons = self._cached_actor_polygons if hasattr(self, '_cached_actor_polygons') else []
        for actor_poly in actor_polygons:
            if robot_footprint.intersects(actor_poly):
                self.metrics_tracker.record_collision()
                print("COLLISION DETECTED!")
                self.running = False
                break
    
    def _render(self):
        """Render simulation frame with realistic warehouse visualization."""
        # Clear screen with simple white background for 2D grid
        self.screen.fill((255, 255, 255))
        
        # Render world grid
        self._render_world()
        
        # Render path
        if self.current_path:
            self._render_path()
        
        # Render actors
        self._render_actors()
        
        # Render robot
        self._render_robot()
        
        # Render LiDAR scan
        self._render_lidar()
        
        # Render HUD
        self._render_hud()
        
        pygame.display.flip()
        
        # Optional: Save frame if recording
        if self.recording:
            pygame.image.save(self.screen, f"videos/frame_{self.frame_count:06d}.png")
    
    def _render_world(self):
        """Render realistic warehouse world with aisles, shelves, and zones."""
        # Calculate visible grid range
        world_left = self.camera_x - self.width / 2 / self.camera_scale
        world_right = self.camera_x + self.width / 2 / self.camera_scale
        world_top = self.camera_y - self.height / 2 / self.camera_scale
        world_bottom = self.camera_y + self.height / 2 / self.camera_scale
        
        # Convert to grid coordinates with margin
        margin = 2.0
        gx_min, gy_min = self.world.world_to_grid(world_left - margin, world_top - margin)
        gx_max, gy_max = self.world.world_to_grid(world_right + margin, world_bottom + margin)
        
        # Clamp to grid bounds
        gx_min = max(0, gx_min - 2)
        gx_max = min(self.world.grid_width, gx_max + 2)
        gy_min = max(0, gy_min - 2)
        gy_max = min(self.world.grid_height, gy_max + 2)
        
        cell_size = max(1, int(self.world.grid_resolution * self.camera_scale))
        
        # SIMPLE 2D GRID - clean, clear grid lines at proper scale
        # Draw ALL grid lines - simple 2D grid visualization
        grid_line_color = (180, 180, 180)  # Light gray for grid lines
        
        # Draw vertical grid lines - EVERY cell boundary
        for x in range(gx_min, gx_max + 1):
            world_x, _ = self.world.grid_to_world(x, 0)
            screen_x = int((world_x - self.camera_x + self.width / 2 / self.camera_scale) * self.camera_scale)
            if 0 <= screen_x < self.width:
                top_screen_y = int((world_top - self.camera_y + self.height / 2 / self.camera_scale) * self.camera_scale)
                bottom_screen_y = int((world_bottom - self.camera_y + self.height / 2 / self.camera_scale) * self.camera_scale)
                pygame.draw.line(self.screen, grid_line_color, 
                               (screen_x, max(0, top_screen_y)),
                               (screen_x, min(self.height, bottom_screen_y)),
                               1)
        
        # Draw horizontal grid lines - EVERY cell boundary
        for y in range(gy_min, gy_max + 1):
            _, world_y = self.world.grid_to_world(0, y)
            screen_y = int((world_y - self.camera_y + self.height / 2 / self.camera_scale) * self.camera_scale)
            if 0 <= screen_y < self.height:
                left_screen_x = int((world_left - self.camera_x + self.width / 2 / self.camera_scale) * self.camera_scale)
                right_screen_x = int((world_right - self.camera_x + self.width / 2 / self.camera_scale) * self.camera_scale)
                pygame.draw.line(self.screen, grid_line_color,
                               (max(0, left_screen_x), screen_y),
                               (min(self.width, right_screen_x), screen_y),
                               1)
        
        # Now draw grid cells with symbols
        if cell_size >= 1:
            for y in range(gy_min, gy_max):
                for x in range(gx_min, gx_max):
                    cell_type = self.world.occupancy_grid[y, x]
                    world_x, world_y = self.world.grid_to_world(x, y)
                    screen_x = int((world_x - self.camera_x + self.width / 2 / self.camera_scale) * self.camera_scale)
                    screen_y = int((world_y - self.camera_y + self.height / 2 / self.camera_scale) * self.camera_scale)
                    
                    if not (0 <= screen_x < self.width and 0 <= screen_y < self.height):
                        continue
                    
                    # SIMPLE 2D GRID - clean colors, no fancy symbols
                    if cell_type == 0:  # Free space - WHITE
                        pygame.draw.rect(self.screen, (255, 255, 255), (screen_x, screen_y, cell_size, cell_size))
                    elif cell_type == 1:  # Wall - BLACK
                        pygame.draw.rect(self.screen, (0, 0, 0), (screen_x, screen_y, cell_size, cell_size))
                    elif cell_type == 2:  # Shelf - BROWN
                        pygame.draw.rect(self.screen, (139, 69, 19), (screen_x, screen_y, cell_size, cell_size))
                    elif cell_type == 3:  # Aisle - LIGHT GRAY
                        pygame.draw.rect(self.screen, (220, 220, 220), (screen_x, screen_y, cell_size, cell_size))
                    elif cell_type == 4:  # Dock - YELLOW
                        pygame.draw.rect(self.screen, (255, 255, 0), (screen_x, screen_y, cell_size, cell_size))
                    elif cell_type == 5:  # Staging - LIGHT BLUE
                        pygame.draw.rect(self.screen, (173, 216, 230), (screen_x, screen_y, cell_size, cell_size))
        
        # Draw inflated obstacles overlay - simple visualization
        if cell_size >= 3:
            for y in range(gy_min, gy_max):
                for x in range(gx_min, gx_max):
                    if self.world.inflated_grid[y, x] == 1 and self.world.occupancy_grid[y, x] == 0:
                        # Draw inflation zone (semi-transparent overlay)
                        world_x, world_y = self.world.grid_to_world(x, y)
                        screen_x = int((world_x - self.camera_x + self.width / 2 / self.camera_scale) * self.camera_scale)
                        screen_y = int((world_y - self.camera_y + self.height / 2 / self.camera_scale) * self.camera_scale)
                        
                        if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                            # Light red overlay for inflation zone
                            s = pygame.Surface((cell_size, cell_size))
                            s.set_alpha(30)
                            s.fill((255, 200, 200))
                            self.screen.blit(s, (screen_x, screen_y))
    
    def _render_path(self):
        """Render planned path with realistic warehouse visualization."""
        if not self.current_path:
            return
        
        points = []
        for wx, wy in self.current_path:
            screen_x = int((wx - self.camera_x + self.width / 2 / self.camera_scale) * self.camera_scale)
            screen_y = int((wy - self.camera_y + self.height / 2 / self.camera_scale) * self.camera_scale)
            points.append((screen_x, screen_y))
        
        if len(points) > 1:
            # Draw path line (thicker, more visible)
            pygame.draw.lines(self.screen, (100, 200, 255), False, points, 3)
            
            # Draw waypoints with different styles
            for i, point in enumerate(points):
                if i == 0:
                    # Start point - green
                    pygame.draw.circle(self.screen, (0, 255, 0), point, 5)
                elif i == len(points) - 1:
                    # Goal point - red, pulsing
                    pulse = int(6 + 3 * np.sin(self.sim_time * 4))
                    pygame.draw.circle(self.screen, (255, 100, 100), point, pulse)
                    pygame.draw.circle(self.screen, (255, 0, 0), point, 6)
                else:
                    # Intermediate waypoints - smaller
                    pygame.draw.circle(self.screen, (100, 200, 255), point, 3)
    
    def _render_actors(self):
        """Render dynamic actors with realistic warehouse appearance."""
        for actor in self.actor_manager.get_actors():
            # Get actor footprint corners
            footprint = actor.get_footprint()
            corners = list(footprint.exterior.coords)[:-1]
            
            # Convert to screen coordinates
            screen_points = []
            center_x, center_y = 0, 0
            for wx, wy in corners:
                screen_x = int((wx - self.camera_x + self.width / 2 / self.camera_scale) * self.camera_scale)
                screen_y = int((wy - self.camera_y + self.height / 2 / self.camera_scale) * self.camera_scale)
                screen_points.append((screen_x, screen_y))
                center_x += screen_x
                center_y += screen_y
            
            center_x //= len(screen_points)
            center_y //= len(screen_points)
            
            if actor.actor_type == 'human':
                # Workers - blue with visibility vest style
                color = (60, 140, 255)  # Safety blue
                pygame.draw.polygon(self.screen, color, screen_points)
                pygame.draw.polygon(self.screen, (30, 80, 200), screen_points, 2)
                # Draw worker symbol (person icon)
                pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), 4)
                # Draw person symbol (head + body)
                pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y - 3), 2)
                pygame.draw.line(self.screen, (255, 255, 255), (center_x, center_y - 1), (center_x, center_y + 3), 2)
            else:
                # Forklifts - yellow/orange industrial
                color = (255, 180, 60)  # Industrial yellow
                pygame.draw.polygon(self.screen, color, screen_points)
                pygame.draw.polygon(self.screen, (200, 120, 40), screen_points, 3)
                # Draw forklift symbol (fork icon)
                pygame.draw.circle(self.screen, (255, 50, 50), (center_x, center_y), 5)
                # Draw fork symbol
                pygame.draw.line(self.screen, (200, 0, 0), (center_x, center_y - 4), (center_x, center_y + 4), 2)
                pygame.draw.line(self.screen, (200, 0, 0), (center_x - 2, center_y + 2), (center_x, center_y + 4), 2)
                pygame.draw.line(self.screen, (200, 0, 0), (center_x + 2, center_y + 2), (center_x, center_y + 4), 2)
    
    def _render_robot(self):
        """Render realistic warehouse robot with task indicators."""
        # Get robot footprint
        footprint = self.robot.get_footprint()
        corners = list(footprint.exterior.coords)[:-1]
        
        # Convert to screen coordinates
        screen_points = []
        for wx, wy in corners:
            screen_x = int((wx - self.camera_x + self.width / 2 / self.camera_scale) * self.camera_scale)
            screen_y = int((wy - self.camera_y + self.height / 2 / self.camera_scale) * self.camera_scale)
            screen_points.append((screen_x, screen_y))
        
        # Draw robot body - SIMPLE 2D shape (no fancy styling)
        robot_color = (0, 0, 255)  # Simple blue
        pygame.draw.polygon(self.screen, robot_color, screen_points)
        pygame.draw.polygon(self.screen, (0, 0, 150), screen_points, 2)  # Darker border
        
        # Draw robot center and orientation
        center_x = int((self.robot.x - self.camera_x + self.width / 2 / self.camera_scale) * self.camera_scale)
        center_y = int((self.robot.y - self.camera_y + self.height / 2 / self.camera_scale) * self.camera_scale)
        
        # Draw orientation arrow - simple and clear
        arrow_len = 0.3 * self.camera_scale
        end_x = center_x + arrow_len * np.cos(self.robot.theta)
        end_y = center_y + arrow_len * np.sin(self.robot.theta)
        pygame.draw.line(self.screen, (255, 0, 0), (center_x, center_y), (int(end_x), int(end_y)), 3)
        
        # Draw robot center circle
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), 2)
        
        # Task indicator overlay (show what robot is doing)
        task_state = self.task_fsm.state.name if hasattr(self.task_fsm.state, 'name') else str(self.task_fsm.state)
        
        # Draw task status indicator above robot
        indicator_y = center_y - 15
        if task_state == 'INSPECT':
            # Inspecting - show inspection indicator
            pygame.draw.circle(self.screen, (255, 200, 0), (center_x, indicator_y), 8)
            pygame.draw.circle(self.screen, (255, 255, 0), (center_x, indicator_y), 5)
        elif task_state == 'GOTO_BAY' or task_state == 'GOTO_DOCK':
            # Moving - show movement indicator
            pygame.draw.circle(self.screen, (0, 255, 100), (center_x, indicator_y), 6)
        elif task_state == 'DONE':
            # Mission complete
            pygame.draw.circle(self.screen, (0, 255, 0), (center_x, indicator_y), 8)
        
        # Draw target waypoint indicator if path exists
        if self.current_path and self.current_waypoint_idx < len(self.current_path):
            waypoint = self.current_path[self.current_waypoint_idx]
            wp_x = int((waypoint[0] - self.camera_x + self.width / 2 / self.camera_scale) * self.camera_scale)
            wp_y = int((waypoint[1] - self.camera_y + self.height / 2 / self.camera_scale) * self.camera_scale)
            
            # Draw pulsing waypoint marker
            pulse = int(5 + 3 * np.sin(self.sim_time * 3))
            pygame.draw.circle(self.screen, (100, 255, 100, 100), (wp_x, wp_y), pulse)
            pygame.draw.circle(self.screen, (0, 255, 0), (wp_x, wp_y), 4)
    
    def _render_lidar(self):
        """Render LiDAR scan with realistic perception visualization."""
        if not hasattr(self, 'last_lidar_ranges') or len(self.last_lidar_ranges) == 0:
            return
        
        ranges = self.last_lidar_ranges
        center_x = int((self.robot.x - self.camera_x + self.width / 2 / self.camera_scale) * self.camera_scale)
        center_y = int((self.robot.y - self.camera_y + self.height / 2 / self.camera_scale) * self.camera_scale)
        
        # Draw LiDAR scan - SIMPLE visualization
        step = max(1, len(ranges) // 60)  # Reduced rays for performance
        for i in range(0, len(ranges), step):
            r = ranges[i]
            if r < self.config['lidar']['max_range']:
                angle = self.robot.theta + self.beam_angles[i] if hasattr(self, 'beam_angles') and i < len(self.beam_angles) else self.robot.theta + np.deg2rad(i * self.config['lidar']['resolution'])
                end_x = center_x + r * self.camera_scale * np.cos(angle)
                end_y = center_y + r * self.camera_scale * np.sin(angle)
                
                # Simple red color for LiDAR rays
                pygame.draw.line(self.screen, (255, 0, 0), (center_x, center_y), (int(end_x), int(end_y)), 1)
    
    def _render_hud(self):
        """Render realistic warehouse robot operational dashboard."""
        # Background panels for HUD
        panel_height = 180
        pygame.draw.rect(self.screen, (20, 20, 30), (0, 0, self.width, panel_height))
        
        # Main title
        title_font = pygame.font.Font(None, 32)
        title_text = title_font.render("Smart Warehouse Robot - Operational Dashboard", True, (255, 255, 255))
        self.screen.blit(title_text, (10, 5))
        
        # Status panel (left side)
        status_font = pygame.font.Font(None, 22)
        status_small = pygame.font.Font(None, 18)
        
        # Mission status
        task_state = self.task_fsm.state.name if hasattr(self.task_fsm.state, 'name') else str(self.task_fsm.state)
        status_color = (100, 255, 100) if task_state == 'DONE' else (255, 200, 100)
        status_text = status_font.render(f"STATUS: {task_state}", True, status_color)
        self.screen.blit(status_text, (10, 35))
        
        # Current task
        current_bay = self.task_fsm.current_bay_idx + 1 if self.task_fsm.current_bay_idx < len(self.task_fsm.bays) else len(self.task_fsm.bays)
        total_bays = len(self.task_fsm.bays)
        task_text = status_small.render(f"Task: Bay {current_bay}/{total_bays} | Planner: {self.planner_type.upper()}", True, (200, 200, 200))
        self.screen.blit(task_text, (10, 60))
        
        # Operational metrics
        metrics = self.metrics_tracker.get_current_metrics()
        path_len = metrics.get('path_length', 0.0) if metrics else 0.0
        min_clear = metrics.get('min_clearance', float('inf')) if metrics else float('inf')
        replans = metrics.get('replans', 0) if metrics else 0
        
        metric_y = 85
        metric_text1 = status_small.render(f"Distance Traveled: {path_len:.2f}m", True, (200, 200, 255))
        self.screen.blit(metric_text1, (10, metric_y))
        metric_y += 20
        metric_text2 = status_small.render(f"Min Clearance: {min_clear:.3f}m", True, (200, 200, 255))
        self.screen.blit(metric_text2, (10, metric_y))
        metric_y += 20
        metric_text3 = status_small.render(f"Re-plans: {replans}", True, (200, 200, 255))
        self.screen.blit(metric_text3, (10, metric_y))
        
        # Performance panel (right side)
        perf_x = self.width - 250
        perf_title = status_font.render("PERFORMANCE", True, (255, 255, 255))
        self.screen.blit(perf_title, (perf_x, 35))
        
        # FPS indicator
        fps = int(self.clock.get_fps())
        fps_color = (100, 255, 100) if fps >= 55 else (255, 200, 100) if fps >= 30 else (255, 100, 100)
        fps_text = status_small.render(f"FPS: {fps}", True, fps_color)
        self.screen.blit(fps_text, (perf_x, 60))
        
        # Time elapsed
        time_text = status_small.render(f"Time: {self.sim_time:.1f}s", True, (255, 255, 255))
        self.screen.blit(time_text, (perf_x, 80))
        
        # Speed indicator - use actual velocity command
        speed = abs(self.last_v) if hasattr(self, 'last_v') else abs(getattr(self.robot, 'v', 0.0))
        speed_color = (100, 255, 100) if speed > 0.1 else (255, 100, 100)
        speed_text = status_small.render(f"Speed: {speed:.2f} m/s", True, speed_color)
        self.screen.blit(speed_text, (perf_x, 100))
        
        # Battery/Energy indicator (simulated)
        energy = max(0, 100 - (self.sim_time / 10))  # Decrease over time
        energy_color = (100, 255, 100) if energy > 50 else (255, 200, 100) if energy > 25 else (255, 100, 100)
        energy_text = status_small.render(f"Energy: {energy:.0f}%", True, energy_color)
        self.screen.blit(energy_text, (perf_x, 120))
        
        # Task progress bar
        if total_bays > 0:
            progress = current_bay / total_bays
            bar_x, bar_y, bar_w, bar_h = 10, 150, self.width - 20, 20
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (100, 200, 255), (bar_x, bar_y, int(bar_w * progress), bar_h))
            progress_text = status_small.render(f"Mission Progress: {int(progress*100)}%", True, (255, 255, 255))
            self.screen.blit(progress_text, (bar_x + 5, bar_y + 2))
    
    def _restart(self):
        """Restart simulation."""
        # Reset robot position
        self.robot.x = 2.0
        self.robot.y = 2.0
        self.robot.theta = 0.0
        
        # Reset FSM
        self.task_fsm.reset()
        
        # Clear path
        self.current_path = None
        self.current_waypoint_idx = 0
        
        # Reset metrics
        self.metrics_tracker.reset()
        
        # Reset time
        self.sim_time = 0.0
        self.frame_count = 0
        
        print("Simulation restarted")
    
    def _cleanup(self):
        """Cleanup and save logs."""
        # Compute final metrics
        final_metrics = self.metrics_tracker.finalize(
            sim_time=self.sim_time,
            success=self.task_fsm.is_done() and not self.metrics_tracker.has_collision()
        )
        
        # Log to CSV
        self.logger.log(**final_metrics)
        
        # Print summary
        print("\n=== Simulation Summary ===")
        print(f"Scenario: {self.scenario}")
        print(f"Planner: {self.planner_type}")
        print(f"Time to goal: {final_metrics.get('t_goal', 0.0):.2f}s")
        print(f"Path length: {final_metrics.get('path_len', 0.0):.2f}m")
        print(f"Min clearance: {final_metrics.get('min_clear', float('inf')):.3f}m")
        print(f"Re-plans: {final_metrics.get('replans', 0)}")
        print(f"Success: {final_metrics.get('success', 0)}")
        print(f"Efficiency: {final_metrics.get('efficiency', 0.0):.2f}")
        
        pygame.quit()

