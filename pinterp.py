"""
Atomic interpolation over periodic boundary conditions
via a boid-inspired scoring function.

Usage: Import the `interp_periodic` function from this file.
Author: Lukas Kim, kim_lukas@berkeley.edu
"""

import numpy as np
from ase import Atoms


def interp_periodic(
    initial: Atoms, final: Atoms, num_images: int = 20, verbose: bool = True
) -> list[Atoms]:
    """Linear interpolation from initial to final. The wrapped and unwrapped paths
    are scored via a [boid](https://en.wikipedia.org/wiki/Boids#Model_details) scoring
    function (lower is better). The function computes scores for three properties that
    approximate the "physicality" of the interpolated trajectory. The relative scores
    of the wrapped and unwrapped trajectories are compared, and the interpolated
    trajectory with the lower score in the most categories is returned.

    Parameters
    ----------
    initial : initial state `ase.Atoms` object
    final : final state `ase.Atoms` object
    num_images : `int` number of interpolation frames
    verbose : `bool` print scores flag
    """
    # create both paths
    linear_path = interp_linear(initial, final, num_images)
    linear_unwrap_path = interp_linear_unwrap(initial, final, num_images)

    # decide: which path looks more natural?
    s1, a1, c1 = boid_score(linear_path)
    s2, a2, c2 = boid_score(linear_unwrap_path)

    # normalize the scores per category then total all scores
    init_scores = np.array([[s1, s2], [a1, a2], [c1, c2]])
    norm_scores = init_scores / np.sum(init_scores, -1, keepdims=True)
    scores = np.sum(norm_scores, 0)

    if verbose:
        print("Wrapped Path:")
        print(f"  separation: {s1:10f}")
        print(f"   alignment: {a1:10f}")
        print(f"    cohesion: {c1:10f}")
        print("Unwrapped Path:")
        print(f"  separation: {s2:10f}")
        print(f"   alignment: {a2:10f}")
        print(f"    cohesion: {c2:10f}")
        print("Final Scores:")
        print("\t            wrapped path unwrapped path")
        print(f"\tseparation: {norm_scores[0, 0]:12f} {norm_scores[0, 1]:14f}")
        print(f"\t alignment: {norm_scores[1, 0]:12f} {norm_scores[1, 1]:14f}")
        print(f"\t  cohesion: {norm_scores[2, 0]:12f} {norm_scores[2, 1]:14f}")

    # compare scores in each category
    linear_total = np.sum(norm_scores[:, 0] < norm_scores[:, 1])
    linear_unwrap_total = 3 - linear_total

    if linear_total > linear_unwrap_total:
        return linear_path
    else:
        # wrap atoms back into the box
        wrapped_path = []
        for frame in linear_unwrap_path:
            frame.wrap()
            wrapped_path.append(frame)
        return wrapped_path


def interp_linear(initial: Atoms, final: Atoms, num_images=20) -> list[Atoms]:
    """Direct linear interpolation from initial to final"""
    images = []
    dR = final.positions - initial.positions
    for t in np.linspace(0, 1, num_images + 2):
        image = initial.copy()
        image.translate(t * dR)
        images.append(image)
    return images


def interp_linear_unwrap(
    initial: Atoms, final: Atoms, num_images=20, rewrap=False
) -> list[Atoms]:
    """Direct linear interpolation from initial to unwrapped final"""
    images = []
    dR = final.positions - initial.positions

    # unwrap translation
    fR = dR @ np.linalg.inv(final.cell)
    offsets = (final.pbc * np.floor(fR + 0.5)) @ final.cell
    dR -= offsets

    for t in np.linspace(0, 1, num_images + 2):
        image = initial.copy()
        image.translate(t * dR)
        if rewrap:
            image.wrap()
        images.append(image)
    return images


def smooth_step(x, scale=10.0, cutoff=2.0):
    """Smooth stepping function"""
    return 1.0 / (1.0 + np.exp(scale * (x - cutoff)))


def boid_score(traj: list[Atoms]):
    """Scoring function based on [boid](https://en.wikipedia.org/wiki/Boids#Model_details)
    movement rules. The three basic rules are (1) separation, (2) alignment, and (3) cohesion.
    In atomic terms, physical trajectories will more closely follow these rules than unphyiscal
    trajectories. In general, atoms in physical trajectories will have fewer collisions (1),
    will tend move in concert with their nearest neighbors (2), and retain molecular shape (3).

    The scores have different magnitudes but the final comparison of the lowest scores in the most
    categories ensures each metric is treated with equal weight.
    """
    nframes, natoms = len(traj), len(traj[0])
    pos = np.array([atoms.positions for atoms in traj])
    vel = pos[1:] - pos[:-1]
    vel_m = np.sqrt(np.sum(vel * vel, -1))

    pos_ij = pos[:, None, :, :] - pos[:, :, None, :]
    Rij = np.sqrt(np.sum(pos_ij * pos_ij, -1))
    Nij = smooth_step(Rij)
    idx = np.arange(natoms)
    Nij[:, idx, idx] = 0.0

    # separation (average number of neighbors within 0.25 angstroms)
    collisions = smooth_step(Rij, cutoff=0.25)
    collisions[:, idx, idx] = 0.0
    collisions = np.sum(collisions, -1)
    separation = np.mean(collisions, 0)

    # alignment (average cosine similarity of interpolation direction with neighbors)
    vel_ij = np.sum((vel[:, :, None, :]) * (vel[:, None, :, :]), -1) / (
        vel_m[:, :, None] * vel_m[:, None, :] + np.finfo(np.float32).eps
    )
    alignment = np.sum((1 - vel_ij) * Nij[:-1], -1) / np.sum(Nij[:-1], -1)
    alignment = np.mean(alignment, 0)

    # cohesion (variance in distance to COM of nearest neighbors)
    masses = traj[0].get_masses()
    neighbor_com = np.sum(
        masses[None, None, :, None] * Nij[:, :, :, None] * pos_ij, -2
    ) / np.sum(masses[None, None, :, None] * Nij[:, :, :, None])
    dn = neighbor_com
    dn_m = np.sqrt(np.sum(dn * dn, -1))
    cohesion = np.var(dn_m, 0)

    # return sum of atom scores
    return np.sum(separation), np.sum(alignment), np.sum(cohesion)


if __name__ == "__main__":
    from ase import io

    print("Case 1: Translation out of frame")
    atoms = io.read("examples/case1.xyz", index=slice(None))
    # io.write("examples/case1-linear.xyz", interp_linear(atoms[0], atoms[1]))
    # io.write(
    #     "examples/case1-linear-unwrap.xyz",
    #     interp_linear_unwrap(atoms[0], atoms[1], rewrap=True),
    # )
    io.write("examples/case1-periodic.xyz", interp_periodic(atoms[0], atoms[1]))
    print()

    print("Case 2: Rotation inside frame")
    atoms = io.read("examples/case2.xyz", index=slice(None))
    # io.write("examples/case2-linear.xyz", interp_linear(atoms[0], atoms[1]))
    # io.write(
    #     "examples/case2-linear-unwrap.xyz",
    #     interp_linear_unwrap(atoms[0], atoms[1], rewrap=True),
    # )
    io.write("examples/case2-periodic.xyz", interp_periodic(atoms[0], atoms[1]))
    print()

    # more frames
    print("Case 1: Translation out of frame")
    atoms = io.read("examples/case1.xyz", index=slice(None))
    io.write(
        "examples/case1-periodic.xyz",
        interp_periodic(atoms[0], atoms[1], num_images=2000),
    )
    print()

    print("Case 2: Rotation inside frame")
    atoms = io.read("examples/case2.xyz", index=slice(None))
    io.write(
        "examples/case2-periodic.xyz",
        interp_periodic(atoms[0], atoms[1], num_images=2000),
    )
    print()
