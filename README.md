# Atomic Interpolation over Periodic Boundary Conditions

This project provides a Python utility to perform interpolation between atomic structures under periodic boundary conditions (PBCs) using a scoring function inspired by [boid](https://en.wikipedia.org/wiki/Boids#Model_details) behavior rules. The goal is to produce a trajectory that mimics physical atomic motion more closely than naive linear interpolation by choosing between the wrapped and unwrapped interpolated paths based on separation, alignment, and cohesion metrics.

## Installation

This package requires:

- [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/)
- NumPy

Install dependencies via pip:

```bash
pip install ase numpy
```

## Usage

Import and use the `interp_periodic` function:

```python
from pinterp import interp_periodic
from ase.io import read, write

# Load two structures from XYZ file
atoms = read("examples/case1.xyz", index=slice(None))

# Interpolate
trajectory = interp_periodic(atoms[0], atoms[1], num_images=20, verbose=True)

# Write interpolated trajectory
write("case1-periodic.xyz", trajectory)
```

## Scoring function

The function `boid_score` assigns a score to a trajectory based on three rules of flocking behavior:

- Separation: Penalizes atoms that are too close (< 0.25 Å)
- Alignment: Rewards similar direction of motion between neighboring atoms
- Cohesion: Measures variance in distance to local centers of mass

The trajectory (wrapped vs unwrapped) with the lowest score in the most categories is chosen.
