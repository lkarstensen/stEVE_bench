from copy import deepcopy
import os
from typing import List, Tuple
import numpy as np
import eve.intervention.vesseltree
from eve.intervention.vesseltree.util.meshing import get_surface_mesh, save_mesh
from eve.intervention.vesseltree.util.voxelcube import (
    create_empty_voxel_cube_from_branches,
    VoxelCube,
)
from eve.intervention.vesseltree.util.branch import BranchWithRadii
from eve.intervention.vesseltree.vesseltree import (
    VesselTree,
    at_tree_end,
    find_nearest_branch_to_point,
)
import skimage.filters


def print_obj_from_selfmade(
    vesseltree: VesselTree,
    struts: List[Tuple[int, int]],
    grid_offset: Tuple[float, float] = [0.0, 0.0],
    z_split: float = None,
    z_remove_lower: float = None,
    z_remove_upper: float = None,
):
    spacing = [0.25, 0.25, 0.25]
    voxel_cube = create_empty_voxel_cube_from_branches(vesseltree, spacing)

    for _ in range(50):
        voxel_cube.add_padding_layer("y", dimension_end="high")
    for _ in range(20):
        voxel_cube.add_padding_layer_all_sides()

    voxel_cube_struts = deepcopy(voxel_cube)

    start_rectangles = []
    strut_start_y = vessel_tree.coordinate_space_episode.high[1]
    strut_start_y += 10.5
    strut_start_y = np.ceil(strut_start_y / spacing[1]) * spacing[1]
    for strut in struts:
        strut_x = strut[0] * 10 + grid_offset[0]
        strut_z = strut[1] * 10 + grid_offset[1]

        strut_xz = np.array([strut_x, strut_z])
        cl_coords = vessel_tree.centerline_coordinates
        cl_coords_xz = np.delete(cl_coords, 1, axis=-1)
        strut_to_cl_xz_dist = np.linalg.norm(cl_coords_xz - strut_xz, axis=-1)
        nearest_cl_idx = np.argmin(strut_to_cl_xz_dist)
        strut_end_y = cl_coords[nearest_cl_idx][1]
        strut_end_y = np.floor(strut_end_y / spacing[1]) * spacing[1]

        strut_end = np.array([strut_x, strut_end_y, strut_z])
        strut_start = np.array([strut_x, strut_start_y, strut_z])

        start_rectangle = mark_first_cylinder(voxel_cube_struts, strut_start)

        start_rectangles.append(start_rectangle)

        start_sec_cyl = start_rectangle + [0, -3.4, 0]

        mark_second_cylinder(voxel_cube_struts, start_sec_cyl, strut_end)

    voxel_cube_struts.gaussian_smooth(1)
    voxel_cube_struts.gaussian_smooth(0.4)
    # voxel_cube_struts.gaussian_smooth(0.4)

    for start_rectangle in start_rectangles:
        mark_rectangle(voxel_cube_struts, start_rectangle)

    for branch in vesseltree:
        voxel_cube.mark_centerline_in_array(
            branch.coordinates, branch.radii, marking_value=1, radius_padding=1.5
        )
        voxel_cube_struts.mark_centerline_in_array(
            branch.coordinates, branch.radii, marking_value=0, radius_padding=0.75
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
    voxel_cube.gaussian_smooth(1)
    voxel_cube.gaussian_smooth(0.7)

    voxel_cube.value_array = np.add(
        voxel_cube.value_array, voxel_cube_struts.value_array
    )

    mesh = get_surface_mesh(voxel_cube, "ascent", level=0.6)
    # mesh = mesh.decimate_pro(0.9)
    mesh = mesh.decimate(0.8)
    # cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_mesh(
        mesh,
        os.path.join(
            dir_path,
            f"aorta_with_struts_type_{vessel_tree.arch_type}_seed_{vessel_tree.seed}_scale_{vessel_tree.scaling_xyzd}_rot_{vessel_tree.rotation_yzx_deg}_omit_{vessel_tree.omit_axis}_full.obj",
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

        mesh = get_surface_mesh(lower_model, "ascent", level=0.6)
        mesh.decimate(0.8, inplace=True)
        save_mesh(
            mesh,
            os.path.join(
                dir_path,
                f"aorta_with_struts_type_{vessel_tree.arch_type}_seed_{vessel_tree.seed}_scale_{vessel_tree.scaling_xyzd}_rot_{vessel_tree.rotation_yzx_deg}_omit_{vessel_tree.omit_axis}_split{z_split}_lower.obj",
            ),
        )

        mesh = get_surface_mesh(upper_model, "ascent", level=0.6)
        mesh.decimate(0.8, inplace=True)
        save_mesh(
            mesh,
            os.path.join(
                dir_path,
                f"aorta_with_struts_type_{vessel_tree.arch_type}_seed_{vessel_tree.seed}_scale_{vessel_tree.scaling_xyzd}_rot_{vessel_tree.rotation_yzx_deg}_omit_{vessel_tree.omit_axis}_split{z_split}_upper.obj",
            ),
        )

    voxel_cube_insertion = get_insertion_voxel_cube(
        vessel_tree, strut_start_y, spacing, grid_offset
    )

    mesh = get_surface_mesh(voxel_cube_insertion, "ascent", level=0.5)
    mesh.decimate(0.8, inplace=True)
    save_mesh(
        mesh,
        os.path.join(
            dir_path,
            f"aorta_with_struts_type_{vessel_tree.arch_type}_seed_{vessel_tree.seed}_scale_{vessel_tree.scaling_xyzd}_rot_{vessel_tree.rotation_yzx_deg}_omit_{vessel_tree.omit_axis}_insertion.obj",
        ),
    )


def mark_first_cylinder(voxel_cube: VoxelCube, strut_start: np.ndarray):
    voxel_coords = voxel_cube.voxel_coords

    dist_to_start = np.linalg.norm(voxel_coords - strut_start, axis=-1)
    start_voxel = np.argmin(dist_to_start)
    start_voxel = np.unravel_index(start_voxel, dist_to_start.shape)

    n_voxels = 2 / voxel_cube.spacing[1]
    n_voxels = int(n_voxels)
    radii = [2.5] * n_voxels
    # radii[0] = radii[-1] = 2.249

    for i in range(n_voxels):
        radius = radii[i]
        idx_y = start_voxel[1] - i

        voxel_idx = (start_voxel[0], idx_y, start_voxel[2])
        in_plane_coords = voxel_coords[:, idx_y, :]
        center_coords = voxel_coords[voxel_idx]
        diff_to_center = in_plane_coords - center_coords
        dist_to_voxel = np.linalg.norm(diff_to_center, axis=-1)
        in_strut = dist_to_voxel <= radius
        idxs_to_mask_x, idxs_to_mask_z = np.where(in_strut)
        idxs_to_mask_y = np.ones_like(idxs_to_mask_x) * idx_y
        voxel_cube.value_array[idxs_to_mask_x, idxs_to_mask_y, idxs_to_mask_z] = 1

    return strut_start + [0, -n_voxels * voxel_cube.spacing[1], 0]


def mark_rectangle(voxel_cube: VoxelCube, rect_start: np.ndarray):
    voxel_coords = voxel_cube.voxel_coords

    dist_to_start = np.linalg.norm(voxel_coords - rect_start, axis=-1)
    start_voxel = np.argmin(dist_to_start)
    start_voxel = np.unravel_index(start_voxel, dist_to_start.shape)

    n_voxels_y = 3.5 / voxel_cube.spacing[1]
    n_voxels_x = 3.25 / voxel_cube.spacing[0]

    x_voxelrange = range(int(-(n_voxels_x - 1) / 2), int((n_voxels_x - 1) / 2) + 1)

    z_distances = [1.75] * int(n_voxels_x)
    z_distances[:3] = [1.0, 1.25, 1.5]
    distances_2 = np.flip(z_distances[:3])
    z_distances[-3:] = distances_2

    for i in range(int(n_voxels_y)):
        idx_y = start_voxel[1] - i
        voxel_idx = (start_voxel[0], idx_y, start_voxel[2])
        dist_to_voxel = np.linalg.norm(
            voxel_coords[:, idx_y, :] - voxel_coords[voxel_idx], axis=-1
        )
        zero_out = dist_to_voxel <= 5
        idxs_to_mask_x, idxs_to_mask_z = np.where(zero_out)
        idxs_to_mask_y = np.ones_like(idxs_to_mask_x) * idx_y
        voxel_cube.value_array[idxs_to_mask_x, idxs_to_mask_y, idxs_to_mask_z] = 0

        for k, j in enumerate(x_voxelrange):
            idx_x = start_voxel[0] + j
            z_dist = z_distances[k]
            voxel_idx = (idx_x, idx_y, start_voxel[2])

            dist_to_voxel = np.linalg.norm(
                voxel_coords[idx_x, idx_y, :] - voxel_coords[voxel_idx], axis=-1
            )

            in_strut = dist_to_voxel <= z_dist
            idxs_to_mask_z = np.where(in_strut)
            idxs_to_mask_x = np.ones_like(idxs_to_mask_z) * idx_x
            idxs_to_mask_y = np.ones_like(idxs_to_mask_z) * idx_y
            voxel_cube.value_array[idxs_to_mask_x, idxs_to_mask_y, idxs_to_mask_z] = 1

    return rect_start + [0, -n_voxels_y * voxel_cube.spacing[1], 0]


def mark_second_cylinder(
    voxel_cube: VoxelCube, strut_start: np.ndarray, strut_end: np.ndarray
):
    voxel_coords = voxel_cube.voxel_coords

    dist_to_start = np.linalg.norm(voxel_coords - strut_start, axis=-1)
    start_voxel = np.argmin(dist_to_start)
    start_voxel = np.unravel_index(start_voxel, dist_to_start.shape)

    n_voxels = abs(strut_start[1] - strut_end[1]) / voxel_cube.spacing[1]
    n_voxels = int(n_voxels)
    radii = [4] * n_voxels
    radii_phase = [3.0, 3.25, 3.5, 3.75]
    radii[:4] = radii_phase
    radii[-4:] = np.flip(radii_phase)

    for i in range(n_voxels):
        radius = radii[i]
        idx_y = start_voxel[1] - i

        voxel_idx = (start_voxel[0], idx_y, start_voxel[2])

        dist_to_voxel = np.linalg.norm(
            voxel_coords[:, idx_y, :] - voxel_coords[voxel_idx], axis=-1
        )
        in_strut = dist_to_voxel <= radius
        idxs_to_mask_x, idxs_to_mask_z = np.where(in_strut)
        idxs_to_mask_y = np.ones_like(idxs_to_mask_x) * idx_y
        voxel_cube.value_array[idxs_to_mask_x, idxs_to_mask_y, idxs_to_mask_z] = 1


def get_insertion_voxel_cube(
    vessel_tree: VesselTree,
    y_start: float,
    spacing: Tuple[float, float, float],
    grid_offset: Tuple[float, float] = [0.0, 0.0],
):
    struts = [[-1, -2], [1, -2]]

    strut_ends = []

    for strut in struts:
        strut_x = strut[0] * 10 + grid_offset[0]
        strut_z = strut[1] * 10 + grid_offset[1]

        strut_end = np.array([strut_x, 0, strut_z])
        strut_ends.append(strut_end)
    radius = 2
    radius_bridge = 4

    aorta_extension = extend_branch_end(vessel_tree["aorta"], "start", 20)
    aorta_extension_clear = extend_branch_end(vessel_tree["aorta"], "start", 10)

    new_radii = np.ones_like(aorta_extension.radii) * radius
    aorta_extension = BranchWithRadii(
        "aorta_extension", aorta_extension.coordinates, new_radii
    )

    strud_dist = np.linalg.norm(strut_ends[0] - strut_ends[1])
    strud_dist = int(np.ceil(strud_dist))
    strud_connection_cl = np.linspace(strut_ends[0], strut_ends[1], strud_dist)
    radii = np.ones((strud_connection_cl.shape[0],)) * radius_bridge
    strud_connection = BranchWithRadii("stud_connection", strud_connection_cl, radii)

    r = 20
    phi = np.linspace(0, np.pi / 3)

    zs = -np.sin(phi) * r
    ys = np.cos(phi) * r - r
    centerline_curve = [[0.0, y, z] for y, z in zip(ys, zs)]
    centerline_curve = np.array(centerline_curve)
    centerline_curve += aorta_extension.coordinates[0]
    radii_curve = np.ones((centerline_curve.shape[0])) * radius

    curve = BranchWithRadii("curve", centerline_curve, radii_curve)

    curve_extension = extend_branch_end(curve, "end", 4)

    voxel_cube = create_empty_voxel_cube_from_branches(
        [
            aorta_extension,
            aorta_extension_clear,
            strud_connection,
            curve,
            curve_extension,
        ],
        spacing,
    )

    for _ in range(50):
        voxel_cube.add_padding_layer("y", dimension_end="high")
    for _ in range(20):
        voxel_cube.add_padding_layer_all_sides()

    voxel_cube_struts = deepcopy(voxel_cube)

    start_rectangles = []
    for strut in struts:
        strut_x = strut[0] * 10 + grid_offset[0]
        strut_z = strut[1] * 10 + grid_offset[1]

        strut_end = np.array([strut_x, 0, strut_z])
        strut_start = np.array([strut_x, y_start, strut_z])

        start_rectangle = mark_first_cylinder(voxel_cube_struts, strut_start)

        start_rectangles.append(start_rectangle)

        start_sec_cyl = start_rectangle + [0, -3.4, 0]

        mark_second_cylinder(voxel_cube_struts, start_sec_cyl, strut_end)

    voxel_cube_struts.gaussian_smooth(1)
    voxel_cube_struts.gaussian_smooth(0.4)
    # voxel_cube_struts.gaussian_smooth(0.4)

    for start_rectangle in start_rectangles:
        mark_rectangle(voxel_cube_struts, start_rectangle)

    voxel_cube.mark_centerline_in_array(strud_connection_cl, radii, marking_value=1)

    voxel_cube.mark_centerline_in_array(
        aorta_extension.coordinates,
        aorta_extension.radii,
        marking_value=1,
        radius_padding=1.5,
    )
    voxel_cube.mark_centerline_in_array(
        curve.coordinates,
        curve.radii,
        marking_value=1,
        radius_padding=1.5,
    )

    voxel_cube.mark_centerline_in_array(
        curve.coordinates,
        curve.radii,
        marking_value=0,
        radius_padding=0,
    )
    voxel_cube.mark_centerline_in_array(
        curve_extension.coordinates,
        curve_extension.radii,
        marking_value=0,
        radius_padding=0,
    )
    voxel_cube.mark_centerline_in_array(
        aorta_extension.coordinates,
        aorta_extension.radii,
        marking_value=0,
        radius_padding=0,
    )
    voxel_cube.mark_centerline_in_array(
        aorta_extension_clear.coordinates,
        aorta_extension.radii,
        marking_value=0,
        radius_padding=2,
    )
    voxel_cube.gaussian_smooth(1)
    voxel_cube.gaussian_smooth(1)
    voxel_cube.gaussian_smooth(1)

    voxel_cube.gaussian_smooth(1)
    voxel_cube.gaussian_smooth(0.7)

    voxel_cube.value_array = np.add(
        voxel_cube.value_array, voxel_cube_struts.value_array
    )
    return voxel_cube


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
        seed=661023725,
        rotation_yzx_deg=[0, -25, 0],
        scaling_xyzd=[0.7526567834727076, 0.7526567834727076, 0.7254665311210199, 0.85],
    )
    vessel_tree.reset()

    print_obj_from_selfmade(
        vessel_tree,
        struts=[
            [0, 0],
            [3, 7],
            [-3, 11],
        ],
        z_split=None,
    )


# eve.intervention.vesseltree.AorticArch(
#             seed=34734120,
#             rotation_yzx_deg=[0, 0, 0],
#             scaling_xyzd=[
#                 1.003423358330579,
#                 1.003423358330579,
#                 0.9375401030776935,
#                 0.85,
#             ],
#         )
