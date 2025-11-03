#!/usr/bin/env python3
"""
Entry point for Smart Warehouse Robot simulation.

Purpose: Parse CLI arguments, load configuration, initialize and run simulation.

Inputs:
    --scenario: Scenario ID (S1, S2, S3, S4)
    --planner: Planner type (a_star, prm)
    --config: Optional config preset (baseline, safe, fast)

Outputs:
    Runs simulation, logs KPIs to CSV, generates output files.

Params:
    scenario: str - Scenario identifier
    planner: str - Global planner to use
    config: str - Configuration preset (default: baseline)
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sim.engine import SimulationEngine


def main():
    """Main entry point for simulation."""
    parser = argparse.ArgumentParser(
        description="Smart Warehouse Robot (CW1) - Pygame Edition"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
        help="Scenario ID: S1 (Baseline), S2 (Crossing Actor), S3 (Blocked Aisle), S4 (Congestion), S5 (Crowded), S6 (Multi-Robot), S7 (Emergency)",
    )
    parser.add_argument(
        "--planner",
        type=str,
        required=True,
        choices=["bfs", "dfs", "dijkstra", "a_star", "rrt", "prm", 
                 "mst", "dfs_spanning", "bfs_spanning", "max_spanning", "rooted_spanning"],
        help="Global planner: bfs, dfs, dijkstra, a_star, rrt, prm, mst, dfs_spanning, bfs_spanning, max_spanning, rooted_spanning",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        choices=["baseline", "safe", "fast"],
        help="Configuration preset (default: baseline)",
    )

    args = parser.parse_args()

    # Ensure data directories exist
    os.makedirs("data/logs", exist_ok=True)
    os.makedirs("docs/img", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    try:
        # Initialize and run simulation
        engine = SimulationEngine(
            scenario=args.scenario,
            planner=args.planner,
            config_preset=args.config,
        )
        engine.run()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

