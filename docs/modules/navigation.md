# Navigation Module

## Map & Representation
- Occupancy grid + inflation radius (C-space safety)
- Grid utilities in `nav/maps.py`
- Resolution sensitivity: keep cell size consistent with footprint
- Shortest-path baseline computed for efficiency metric

## Global Planners
- **A\*** (`nav/a_star.py`): grid-based, octile heuristic, early-exit, path smoothing
- **PRM or RRT\*** (`nav/prm.py`): sampling-based roadmap/tree; collision checks with inflation
- Both return path + stats (runtime, path length)

## Local Avoider
- DWA-like controller in `nav/local_dwa.py`
- Velocity sampling with scoring: w_goal*progress - w_obs*(1/min_clear) - w_smooth*Δu
- Inputs: lidar ranges, current path; Output: velocity commands (v, ω)
- Re-plan trigger if path blocked or min-clearance < τ

## Task FSM
- Task executive in `nav/task_fsm.py`
- States: `IDLE → GOTO(bay) → INSPECT → GOTO(dock) → DONE`
- Recovery: wait, turn-in-place, choose alternate aisle
- Deadlock detection and timeout handling

## Config presets
- `config/baseline.yaml` (balanced), `config/safe.yaml` (larger inflation, slower), `config/fast.yaml` (smaller inflation, faster)
