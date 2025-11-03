# Checklists

## Setup
- [ ] Virtual environment created and dependencies installed
- [ ] `python run.py --scenario S1 --planner a_star` starts robot in world
- [ ] `python run.py --scenario S2 --planner a_star` spawns actors (seeded)

## World/Robot
- [ ] Shelves, aisles, dock modeled in grid/tile map
- [ ] LiDAR raycast returns ranges; footprint configured
- [ ] Max speeds realistic; kinematic model respects limits

## Planning/Control
- [ ] A\* (`nav/a_star.py`) returns valid path on inflated grid
- [ ] PRM/RRT\* (`nav/prm.py`) returns valid alternative path
- [ ] Local avoider (`nav/local_dwa.py`) prevents collisions with movers
- [ ] Re-plan fires on blockage (S3 scenario)

## Scenarios
- [ ] S1–S4 runnable via `python run.py --scenario SX --planner Y`
- [ ] CSV logs created in `data/logs/` with all KPI fields
- [ ] Plots generated in `docs/img/` from CSV logs

## Docs & Deliverables
- [ ] README quickstart works on fresh clone
- [ ] Module docs filled & linked
- [ ] Videos recorded (15–45s each) in `videos/`
- [ ] Report uses tables/charts from logs
- [ ] Slides include demo clips + key results
