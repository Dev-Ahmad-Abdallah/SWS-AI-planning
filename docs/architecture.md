# Architecture

The system is built on Pygame for 2D simulation:

```mermaid
flowchart LR
  Goals[Goals: Bay list] -->|targets| FSM[nav/task_fsm.py]
  FSM -->|waypoints| Global{nav/a_star.py<br/>or nav/prm.py}
  Global -->|path| Local[nav/local_dwa.py]
  Local -->|v, ω| Robot[robot/kinematics.py]
  Robot -->|lidar| Sensors[robot/sensors.py]
  Sensors -->|ranges, min_clear| Local
  World[sim/world.py<br/>+ sim/actors.py] -->|obstacles| Sensors
  Sensors -->|blocked/near| Events[Replan Trigger]
  Events --> Global
  Engine[sim/engine.py<br/>Pygame 60 FPS] -->|renders| World
```

## Component Modules

- **sim/engine.py**: Pygame main loop (60 FPS), rendering layers, camera panning, pause/record toggles
- **sim/world.py**: Grid/tile map loader, occupancy grid, inflation computation
- **sim/actors.py**: Dynamic obstacle spawner with seeded RNG
- **robot/kinematics.py**: Diff-drive/unicycle model with speed limits
- **robot/sensors.py**: 2D LiDAR raycast, min-clearance monitor
- **nav/a_star.py**: Grid-based A* planner
- **nav/prm.py**: Sampling-based PRM planner
- **nav/local_dwa.py**: DWA-like local avoidance
- **nav/task_fsm.py**: Task executive with state machine
- **nav/maps.py**: Map utilities, shortest-path baseline
- **tools/logger.py**: CSV logging per run
- **tools/metrics.py**: KPI computation (efficiency, success, etc.)
- **tools/plots.py**: Matplotlib charts for comparison
