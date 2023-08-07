from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import eve.intervention.vesseltree

from eve.intervention.vesseltree.vesseltree import VesselTree
from eve.intervention.vesseltree import BranchWithRadii


def get_strut_pos(
    vesseltree: VesselTree, grid_offset: Tuple[float, float] = [0.0, 0.0]
):
    vessel_x = [branch.coordinates[:, 0] for branch in vessel_tree.branches]
    vessel_z = [branch.coordinates[:, 2] for branch in vessel_tree.branches]

    min_x = np.min(np.concatenate(vessel_x))
    min_x = np.floor(min_x / 10) * 10 - grid_offset[0]
    max_x = np.max(np.concatenate(vessel_x))

    min_z = np.min(np.concatenate(vessel_z))
    min_z = np.floor(min_z / 10) * 10 - grid_offset[1]

    max_z = np.max(np.concatenate(vessel_z))

    grid_x, grid_z = np.meshgrid(
        np.arange(min_x, max_x + 10, 10), np.arange(min_z - 20, max_z + 10, 10)
    )

    ax = plt.gca()

    for branch in vessel_tree.branches:
        for point, radius in zip(branch.coordinates, branch.radii):
            point = np.delete(point, 1)
            circle = plt.Circle(point, radius, color="r")
            ax.add_patch(circle)

    for line_x, line_z in zip(grid_x, grid_z):
        for x, y in zip(line_x, line_z):
            point = np.array([x, y])
            circle = plt.Circle(point, 4, color="b")
            ax.add_patch(circle)

    ax.scatter(grid_x, grid_z)

    plt.show()


if __name__ == "__main__":
    vessel_tree = eve.intervention.vesseltree.AorticArch(
        seed=661023725,
        rotation_yzx_deg=[0, -25, 0],
        scaling_xyzd=[0.7526567834727076, 0.7526567834727076, 0.7254665311210199, 0.85],
    )
    vessel_tree.reset()

    get_strut_pos(vessel_tree)
