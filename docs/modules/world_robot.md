# World & Robot Module

## World
- Grid/tile map with aisles (≥ 3), shelves, dock, service corridor
- Loaded from ASCII/PNG layout into numpy grid in `sim/world.py`
- Occupancy grid with inflation via Minkowski sum (shapely) using robot footprint
- Dynamic actors: human + forklift spawned in `sim/actors.py`
- Random seeds per scenario; configurable density via `config/actors.yaml`

## Robot
- Diff-drive/unicycle kinematic model in `robot/kinematics.py`
- Rectangular footprint with inflation radius (C-space representation)
- Speed limits: v_max, ω_max enforced
- Sensors: 2D LiDAR raycast in `robot/sensors.py` (required)
- Min-clearance monitor triggers re-planning at 10–20 Hz

## Running
- `python run.py --scenario S1 --planner a_star` → static world + robot
- `python run.py --scenario S2 --planner a_star` → + actors + logger
