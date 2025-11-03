# Smart Warehouse Robot (CW1)

Simulated warehouse robot that navigates aisles, handles **dynamic obstacles**, and compares **A\*** to **PRM/RRT\***. Includes KPI logging and plots for the report.

## Quickstart
```bash
# Setup virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

# Run scenarios (Pygame simulator)
python run.py --scenario S1 --planner a_star    # S1 Nominal
python run.py --scenario S2 --planner a_star    # S2 Crossing actor
python run.py --scenario S3 --planner prm       # S3 Blocked aisle → re-plan
python run.py --scenario S4 --planner a_star    # S4 Congestion

# Generate plots from logs
python tools/plots.py --in data/logs/*.csv --out docs/img/
```

## Scenarios
- **S1** Nominal (static world, no dynamic obstacles)
- **S2** Crossing actor (human/vehicle crosses path)
- **S3** Blocked aisle → re-planning required
- **S4** Congestion (multi-actor environment)

## Structure
- `sim/` - Pygame engine, world grid, dynamic actors
- `robot/` - Kinematics model, LiDAR sensors
- `nav/` - Global planners (A*, PRM/RRT*), local avoidance, task FSM, map utilities
- `tools/` - Logging, metrics calculation, plotting
- `config/` - YAML configuration files (baseline, safe, fast, actors)
- `data/logs/` - CSV logs per run
- `docs/` - Documentation and report materials
- `videos/` - Recorded simulation demos

## Outputs

- CSV logs → `data/logs/`
- Plots → `docs/img/`
- Demo videos → `videos/`

## Requirements

- Handles moving obstacles + re-plan
- Compare two global planners (A* vs PRM/RRT*)
- Log KPIs; present results with charts



---


