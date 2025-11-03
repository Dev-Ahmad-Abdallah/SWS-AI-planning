"""
Robot kinematics model.

Purpose: Unicycle/diff-drive kinematics, state integration, collision checking.

Inputs:
    - Velocity commands (v, omega)
    - Time step (dt)
    - World for collision checks

Outputs:
    - Updated robot state (x, y, theta)
    - Robot footprint polygon for collision detection

Params:
    x, y: float - Initial position
    theta: float - Initial orientation
    v_max: float - Maximum forward velocity
    omega_max: float - Maximum angular velocity
    width: float - Robot width
    length: float - Robot length
"""

import numpy as np
from shapely.geometry import Polygon, box


class RobotKinematics:
    """Unicycle/diff-drive robot kinematics."""
    
    def __init__(self, x=0.0, y=0.0, theta=0.0, v_max=1.0, omega_max=1.0,
                 width=0.5, length=0.7):
        """
        Initialize robot kinematics.
        
        Args:
            x, y: Initial position
            theta: Initial orientation (radians)
            v_max: Maximum forward velocity
            omega_max: Maximum angular velocity
            width: Robot width
            length: Robot length
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.v_max = v_max
        self.omega_max = omega_max
        self.width = width
        self.length = length
        
        # Current velocities
        self.v = 0.0
        self.omega = 0.0
    
    def update(self, v_cmd, omega_cmd, dt, world):
        """
        Update robot state with velocity commands.
        
        Args:
            v_cmd: Commanded forward velocity
            omega_cmd: Commanded angular velocity
            dt: Time step
            world: World object for collision checking
        """
        # Clamp velocities to limits
        self.v = np.clip(v_cmd, -self.v_max, self.v_max)
        self.omega = np.clip(omega_cmd, -self.omega_max, self.omega_max)
        
        # Predict new state
        new_theta = self.theta + self.omega * dt
        
        # Unicycle model
        new_x = self.x + self.v * np.cos(new_theta) * dt
        new_y = self.y + self.v * np.sin(new_theta) * dt
        
        # CRITICAL FIX: Check entire robot footprint for collisions, not just center point
        new_footprint = self._get_footprint_at(new_x, new_y, new_theta)
        
        # Check if robot footprint intersects obstacles
        # Check multiple points on the footprint boundary
        footprint_center = (new_x, new_y)
        corners = list(new_footprint.exterior.coords)[:-1]  # Get all corners
        
        # Check center and all corners
        check_points = [footprint_center] + corners
        
        # Check if any point is in an obstacle
        collision_detected = False
        for px, py in check_points:
            if not world.is_free(px, py, use_inflated=True):
                collision_detected = True
                break
        
        # Also check intermediate points along the footprint edges for more thorough checking
        if not collision_detected:
            # Sample points along edges
            for i in range(len(corners)):
                p1 = corners[i]
                p2 = corners[(i + 1) % len(corners)]
                # Check 3 intermediate points along edge
                for t in [0.25, 0.5, 0.75]:
                    mid_x = p1[0] + t * (p2[0] - p1[0])
                    mid_y = p1[1] + t * (p2[1] - p1[1])
                    if not world.is_free(mid_x, mid_y, use_inflated=True):
                        collision_detected = True
                        break
                if collision_detected:
                    break
        
        # Only update if NO collision detected
        if not collision_detected:
            self.x = new_x
            self.y = new_y
            self.theta = new_theta
        else:
            # STOP if collision would occur - don't move into obstacle
            # Keep current position but zero velocity
            self.v = 0.0
            self.omega = 0.0
    
    def _get_footprint_at(self, x, y, theta):
        """Get robot footprint at given position and orientation."""
        # Create rectangle centered at origin
        half_length = self.length / 2
        half_width = self.width / 2
        
        # Corners of rectangle (local frame)
        corners_local = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])
        
        # Rotate and translate
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        corners_rotated = (rotation_matrix @ corners_local.T).T
        corners_world = corners_rotated + np.array([x, y])
        
        return Polygon(corners_world)
    
    def get_footprint(self):
        """Get current robot footprint as shapely polygon."""
        return self._get_footprint_at(self.x, self.y, self.theta)
    
    def get_state(self):
        """Get current robot state."""
        return (self.x, self.y, self.theta)
    
    def set_state(self, x, y, theta):
        """Set robot state."""
        self.x = x
        self.y = y
        self.theta = theta

