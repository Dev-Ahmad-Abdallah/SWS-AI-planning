# World & Robot

## World Representation

The warehouse world is implemented as a 2D grid/tile map in `sim/world.py`:
- Grid loaded from ASCII or PNG layout files
- Contains aisles (≥ 3), shelves, dock area, and service corridors
- Occupancy grid representation for path planning
- Configuration-space (C-space) inflation via Minkowski sum using shapely
  - Robot footprint is rectangular, inflated to ensure collision-free paths
  - Inflation radius configurable via `config/baseline.yaml`, `config/safe.yaml`, `config/fast.yaml`

## Robot Model

The robot is modeled in `robot/kinematics.py`:
- **Kinematics**: Unicycle/diff-drive model
  - State integration with deterministic timestep (dt)
  - Velocity limits: v_max, ω_max enforced via clamping
  - Position and orientation updated per frame
- **Footprint**: Rectangular shape defined by width and length
  - Collision checks performed via shapely polygon against inflated grid
  - No tunneling through walls or obstacles
- **Motion**: PID speed control (if implemented) or direct velocity commands

## Sensors

LiDAR sensor implemented in `robot/sensors.py`:
- **2D LiDAR raycast**: Field-of-view, resolution, and range configurable
- Raycast performed against world polygons (inflated obstacles)
- Returns range measurements per beam
- **Min-clearance monitor**: Computes minimum distance to obstacles at 10–20 Hz
  - Used to trigger re-planning when clearance drops below threshold τ
  - Critical for dynamic obstacle avoidance

## Dynamic Actors

Dynamic obstacles (humans, forklifts) implemented in `sim/actors.py`:
- Spawned with configurable count, speed ranges per scenario
- Behavior: lane-follow or random walk patterns
- Deterministic RNG seeds per scenario for reproducible runs
- Configuration via `config/actors.yaml`
- Actors avoid walls and can trigger crossing events

## Running Simulations

- Static world: `python run.py --scenario S1 --planner a_star`
- With dynamic actors: `python run.py --scenario S2 --planner a_star`
- Blocked aisle (re-plan test): `python run.py --scenario S3 --planner prm`
- Congestion: `python run.py --scenario S4 --planner a_star`

