# Testing & Evaluation Module

## Scenarios (scripted)
- **S1** Nominal (static)
- **S2** Crossing actor
- **S3** Blocked aisle → re-plan
- **S4** Congestion (multi-actor)

## KPIs (per run → CSV)
- `t_goal`, `path_len`, `min_clear`, `replans`, `cpu_ms`, `success`
- Efficiency = `path_len / shortest_len`

## Plots
- Comparison: A\* vs PRM/RRT\* (bar/line)
- Ablation: inflation radius / local-avoid params vs safety & efficiency
