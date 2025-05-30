import numpy as np
from ase import io, Atoms, build


def create_case1():
    """Case 1: molecule drifts into adjacent cell"""
    atoms_initial = build.molecule("CH4")
    atoms_initial.set_cell([5, 5, 5])
    atoms_initial.set_pbc([True, True, True])
    atoms_initial.center()

    atoms_final = atoms_initial.copy()
    atoms_final.translate([2, 0, 0])
    atoms_final.wrap()
    io.write("examples/case1.xyz", [atoms_initial, atoms_final], format="extxyz")


def create_case2():
    """Case 2: molecule rotates from one end to the other"""
    atoms_initial = build.molecule("CH4")
    atoms_initial.set_cell([5, 5, 5])
    atoms_initial.set_pbc([True, True, True])
    atoms_initial.center()
    d = 0.5 * (2.5 * np.sqrt(3) / 2)
    atoms_initial.translate([d, d, d])

    atoms_final = atoms_initial.copy()
    atoms_final.euler_rotate(180, 0, 0, center=[2.5, 2.5, 2.5])
    atoms_final.wrap()
    io.write("examples/case2.xyz", [atoms_initial, atoms_final], format="extxyz")


def create_case3():
    """Case 3: random to random positions"""
    atoms_initial = build.molecule("CH4")
    atoms_initial.set_cell([5, 5, 5])
    atoms_initial.set_pbc([True, True, True])
    atoms_initial.positions = np.random.uniform(0, 5, (len(atoms_initial), 3))
    atoms_initial.center()

    atoms_final = atoms_initial.copy()
    atoms_final.positions = np.random.uniform(0, 5, (len(atoms_initial), 3))
    io.write("examples/case3.xyz", [atoms_initial, atoms_final], format="extxyz")


if __name__ == "__main__":
    create_case1()
    create_case2()
    create_case3()
