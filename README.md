# Atomic Interpolation over Periodic Boundary Conditions

This project performs linear interpolation between atomic structures under periodic boundary conditions (PBCs) using a scoring function inspired by [boid](https://en.wikipedia.org/wiki/Boids#Model_details) behavior rules. When linearly interpolating between two geometries in a periodic cell, there are two directions on which the geometries can be transformed, either by staying within the unit cell or by going through the boundary. Typically, only one of the two is a physical transformation. Without chemical bond information, the physically correct transformation can only be determined by visual inspection. The goal of this project is to determine the more physical transformation via heuristics by choosing between the wrapped and unwrapped interpolated paths based on separation, alignment, and cohesion metrics.

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

- Separation: Average of number of neighbors within < 0.25 Å over the trajectory
- Alignment: Average of total cosine similarity between the interpolation step direction of nearest neighbors
- Cohesion: Variance of the distance beween each atom and the center of mass of its neighbors

The trajectory (wrapped vs unwrapped) with the lowest score in the most categories is chosen.
