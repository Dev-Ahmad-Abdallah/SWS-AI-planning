# Overview

**Objective.** Build a simulated warehouse robot using Pygame that:
1) navigates shelves/aisles to target bays and dock,
2) handles **dynamic obstacles** safely (no collisions),
3) implements both **discrete (A\*)** and **sampling (PRM/RRT\*)** global planning,
4) logs KPIs and presents a brief **comparative study**.

**Success criteria.**
- Zero collisions in S1–S3; bounded wait in S4
- Re-plan on blockage within threshold latency
- KPI tables + plots used in the report and slides
- Pygame simulation runs at stable 60 FPS
- Deterministic runs with seeded RNG per scenario