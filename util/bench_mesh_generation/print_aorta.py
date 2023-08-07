import os
import numpy as np
import eve.intervention.vesseltree
from eve.intervention.vesseltree.util.meshing import get_surface_mesh, save_mesh
from eve.intervention.vesseltree.util.voxelcube import (
    create_empty_voxel_cube_from_branches,
    VoxelCube,
)
from eve.intervention.vesseltree.util.branch import BranchWithRadii
from eve.intervention.vesseltree.vesseltree import VesselTree, at_tree_end


def print_obj_from_selfmade(
    vesseltree: VesselTree,
    z_split: float = None,
    z_remove_lower: float = None,
    z_remove_upper: float = None,
):
    voxel_cube = create_empty_voxel_cube_from_branches(vesseltree, [0.3, 0.3, 0.3])
    for _ in range(100):
        voxel_cube.add_padding_layer_all_sides()

    for branch in vesseltree:
        voxel_cube.mark_centerline_in_array(
            branch.coordinates, branch.radii, marking_value=1, radius_padding=1.5
        )
    for branch in vesseltree:
        voxel_cube.mark_centerline_in_array(
            branch.coordinates, branch.radii, marking_value=0, radius_padding=0
        )

    end_extensions = []
    for branch in vesseltree:
        start = branch.coordinates[0]
        if at_tree_end(start, vessel_tree):
            new_branch = extend_branch_end(branch, "start", 20)
            end_extensions.append(new_branch)

        end = branch.coordinates[-1]
        if at_tree_end(end, vessel_tree):
            new_branch = extend_branch_end(branch, "end", 20)
            end_extensions.append(new_branch)
    for branch in end_extensions:
        radius = 4 if branch.name == "aorta" else 2
        voxel_cube.mark_centerline_in_array(
            branch.coordinates, marking_value=0, cl_radii=radius
        )

    voxel_cube.gaussian_smooth(1)
    voxel_cube.gaussian_smooth(1)
    voxel_cube.gaussian_smooth(1)

    # voxel_cube.gaussian_smooth(1)
    voxel_cube.gaussian_smooth(1.0)
    voxel_cube.gaussian_smooth(0.7)

    mesh = get_surface_mesh(voxel_cube, "ascent")
    # mesh = mesh.decimate_pro(0.9)
    mesh = mesh.decimate(0.95)
    # cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_mesh(
        mesh,
        os.path.join(
            dir_path,
            f"aorta_type_{vessel_tree.arch_type}_seed_{vessel_tree.seed}_scale_{vessel_tree.scaling_xyzd}_rot_{vessel_tree.rotation_yzx_deg}_omit_{vessel_tree.omit_axis}_full.obj",
        ),
    )
    if z_split is not None:
        z_split_idx = int(z_split / voxel_cube.spacing[2])

        lower_model = VoxelCube(
            voxel_cube.value_array.copy(),
            voxel_cube.spacing,
            voxel_cube.world_offset,
        )
        lower_model.value_array[:, :, z_split_idx:] = 0
        if z_remove_lower is not None:
            z_remove_lower_idx = int(z_remove_lower / voxel_cube.spacing[2])
            lower_model.value_array[:, :, :z_remove_lower_idx] = 0

        upper_model = VoxelCube(
            voxel_cube.value_array.copy(),
            voxel_cube.spacing,
            voxel_cube.world_offset,
        )
        upper_model.value_array[:, :, :z_split_idx] = 0
        if z_remove_upper is not None:
            z_remove_upper_idx = int(z_remove_upper / voxel_cube.spacing[2])
            lower_model.value_array[:, :, z_remove_upper_idx:] = 0

        mesh = get_surface_mesh(lower_model, "ascent")
        mesh.decimate(0.95, inplace=True)
        save_mesh(
            mesh,
            os.path.join(
                dir_path,
                f"aorta_type_{vessel_tree.arch_type}_seed_{vessel_tree.seed}_scale_{vessel_tree.scaling_xyzd}_rot_{vessel_tree.rotation_yzx_deg}_omit_{vessel_tree.omit_axis}_split{z_split}_lower.obj",
            ),
        )

        mesh = get_surface_mesh(upper_model, "ascent")
        mesh.decimate(0.95, inplace=True)
        save_mesh(
            mesh,
            os.path.join(
                dir_path,
                f"aorta_type_{vessel_tree.arch_type}_seed_{vessel_tree.seed}_scale_{vessel_tree.scaling_xyzd}_rot_{vessel_tree.rotation_yzx_deg}_omit_{vessel_tree.omit_axis}_split{z_split}_upper.obj",
            ),
        )


def extend_branch_end(branch: BranchWithRadii, start_end: str, length: int):
    if start_end == "start":
        coord_idx = 0
        direction_idx = 1
    else:
        coord_idx = -1
        direction_idx = -2
    coord_start = branch.coordinates[coord_idx]
    direction = branch.coordinates[coord_idx] - branch.coordinates[direction_idx]
    direction = direction / np.linalg.norm(direction)
    coord_end = coord_start + length * direction

    radius = branch.radii[coord_idx]

    new_points = np.linspace(
        coord_end, coord_start, num=int(np.ceil(length)), endpoint=False
    )
    if not start_end == "start":
        new_points = np.flip(new_points, axis=0)

    n_points = new_points.shape[0]

    new_radii = np.ones([n_points]) * radius

    return BranchWithRadii(branch.name, new_points, new_radii)


if __name__ == "__main__":
    vessel_tree = eve.intervention.vesseltree.AorticArch(
        seed=1669028594,
        rotation_yzx_deg=[0, -25, 0],
        scaling_xyzd=[0.9995400602406534, 0.9995400602406534, 0.8178605960609021, 0.85],
    )
    vessel_tree.reset()

    print_obj_from_selfmade(vessel_tree, z_split=100)
