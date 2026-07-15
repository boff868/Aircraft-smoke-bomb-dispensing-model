# Aircraft Smoke Bomb Dispensing Model

Python simulation and optimization scripts for an aircraft/UAV smoke-bomb dispensing problem. The code models missile trajectories, UAV flight paths, smoke-cloud motion, target geometry, and the effective shielding time produced by different smoke-bomb release strategies.

## What This Project Does

- Simulates missile movement toward a target.
- Models UAV smoke-bomb release, free-fall delay, detonation position, and cloud sinking.
- Checks whether the smoke cloud blocks the line of sight between missile and target key points.
- Searches for better dispensing strategies using numerical simulation and optimization.
- Covers single-UAV, multi-bomb, multi-UAV, and multi-missile style scenarios across the question scripts.

## Main Files

```text
question2.py       # Single-UAV strategy objective and optimization setup
question2_2.py     # Alternative geometry/sampling implementation for question 2
question3*.py      # Multi-bomb / target key-point coverage simulations
question4.py       # Multi-UAV strategy simulation
question5-6.py     # Multi-missile and multi-UAV allocation model
question5_5.py     # Effective coverage calculation for a planned deployment
q4enhanced.py      # Enhanced optimization workflow using candidate search
```

## Requirements

The scripts use common scientific Python packages:

```bash
pip install numpy scipy pandas tqdm
```

Some scripts only need `numpy` and `tqdm`; `q4enhanced.py` also imports `pandas` and `scipy`.

## How to Run

Run an individual question script from the repository root:

```bash
python question2.py
python question4.py
python question5-6.py
```

Each file is mostly self-contained and represents a different stage or variant of the mathematical model.

## Notes

The variable names and comments follow the original modeling problem statement. The scripts are intended for numerical experimentation and report/model development, not as a packaged Python library.
