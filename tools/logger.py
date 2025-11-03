"""
CSV logger for KPI logging.

Purpose: Log KPIs per run to CSV files with schema matching requirements.

Inputs:
    - Scenario ID
    - Planner type
    - KPI metrics

Outputs:
    - CSV file in data/logs/ with all required fields

Params:
    scenario: str - Scenario identifier
    planner: str - Planner type
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict


class KPILogger:
    """CSV logger for KPIs."""
    
    def __init__(self, scenario: str, planner: str):
        """
        Initialize KPI logger.
        
        Args:
            scenario: Scenario ID (S1, S2, S3, S4)
            planner: Planner type (a_star, prm)
        """
        self.scenario = scenario
        self.planner = planner
        
        # Ensure logs directory exists
        os.makedirs("data/logs", exist_ok=True)
        
        # CSV schema
        self.csv_schema = [
            "scenario",
            "planner",
            "t_goal",
            "path_len",
            "min_clear",
            "replans",
            "cpu_ms",
            "success",
            "efficiency"
        ]
    
    def log(self, **metrics):
        """
        Log metrics to CSV.
        
        Args:
            **metrics: Dictionary with KPI values
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/logs/{self.scenario}_{self.planner}_{timestamp}.csv"
        
        # Prepare row
        row = {
            "scenario": self.scenario,
            "planner": self.planner,
            "t_goal": metrics.get("t_goal", 0.0),
            "path_len": metrics.get("path_len", 0.0),
            "min_clear": metrics.get("min_clear", float('inf')),
            "replans": metrics.get("replans", 0),
            "cpu_ms": metrics.get("cpu_ms", 0.0),
            "success": metrics.get("success", 0),
            "efficiency": metrics.get("efficiency", 0.0)
        }
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(filename)
        
        # Write CSV
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_schema)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
        
        print(f"Logs saved to {filename}")

