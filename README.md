# UR5 Robot Path and Trajectory Tracking Experiments

## Introduction

This project implements optimal control formulations for path and trajectory tracking on a UR5 robotic arm. It explores four progressive experiments that increase in complexity, starting from basic path tracking without terminal costs, through cyclic path tracking with terminal costs, to trajectory tracking with hard and soft constraints, and finally minimum-time optimal control with variable time steps.

The experiments use CasADi for optimal control problem formulation and solving, with visualization and result plotting capabilities.


## Running the Experiments
**Q1: Path Tracking without Terminal Cost**
```bash
python main.py --experiment Q1
```
Basic path tracking that minimizes velocity, acceleration, and path tracking errors without returning to the initial state.

**Q2: Path Tracking with Cyclic Terminal Cost**
```bash
python main.py --experiment Q2
```
Path tracking with an additional cyclic terminal cost that enforces the robot to return to its initial configuration at the end.

**Q3: Cyclic Trajectory Tracking (Hard & Soft Constraints)**
```bash
python main.py --experiment Q3
```
Trajectory tracking with cyclic constraints, comparing hard constraint formulations against soft constraint formulations.

**Q4: Minimum Time Path & Trajectory Tracking**
```bash
python main.py --experiment Q4
```
Advanced minimum-time optimal control with variable time steps, optimizing both path/trajectory tracking and motion duration.

### Additional Options

Disable visualization during execution:
```bash
python main.py --experiment Q1 --no-viz
```

Disable plotting of results:
```bash
python main.py --experiment Q1 --no-plot
```

Combine options:
```bash
python main.py --experiment all --no-viz --no-plot
```

## Project Structure

```graphql
├── main.py                      # Main entry point and experiment orchestration
├── A2_2025.pdf                  # Assignement pdf
├── A2_report.pdf                # Report with the answare to the assignment
├── robot_setup.py               # UR5 robot configuration and setup
├── experiments/                 # Experiment implementations
│   ├── Q1.py                       # Path tracking without terminal cost
│   ├── Q2.py                       # Path tracking with cyclic terminal cost
│   ├── Q3.py                       # Trajectory tracking with hard/soft constraints
│   └── Q4.py                       # Minimum time path & trajectory tracking
├── utility/                     # Utility functions and helpers
│   ├── visualization.py            # Robot motion visualization
│   ├── plotting.py                 # Result plotting functions
│   └── utility.py                  # Common utilities and solution extraction
├── results/                     # Experiment results and summaries
│   ├── Q1/
│   ├── Q2/
│   ├── Q3/
│   └── Q4/
└── README.md                    # This file
```
