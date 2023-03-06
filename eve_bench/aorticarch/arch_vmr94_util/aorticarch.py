from dataclasses import dataclass
from typing import Tuple, Union
from xml.dom import minidom
from tempfile import gettempdir
import os
import numpy as np
import pyvista as pv
import gymnasium as gym
from .branch import Branch, calc_branching, rotate
from .vmrdownload import download_vmr_files

SCALING_FACTOR = 10
LOW_HIGH_BUFFER = 3


@dataclass
class Insertion:
    position: np.ndarray
    direction: np.ndarray


def _load_points_from_pth(pth_file_path: str, vtu_mesh: pv.UnstructuredGrid) -> Branch:
    points = []
    name = os.path.basename(pth_file_path)[:-4]
    with open(pth_file_path, "r", encoding="utf-8") as file:
        next(file)
        next(file)
        tree = minidom.parse(file)

    xml_points = tree.getElementsByTagName("pos")

    points = []
    radii = []
    low = np.array([vtu_mesh.bounds[0], vtu_mesh.bounds[2], vtu_mesh.bounds[4]])
    high = np.array([vtu_mesh.bounds[1], vtu_mesh.bounds[3], vtu_mesh.bounds[5]])
    low += LOW_HIGH_BUFFER
    high -= LOW_HIGH_BUFFER
    for point in xml_points:
        x = float(point.attributes["x"].value) * SCALING_FACTOR
        y = float(point.attributes["y"].value) * SCALING_FACTOR
        z = float(point.attributes["z"].value) * SCALING_FACTOR
        if np.any([x, y, z] < low) or np.any([x, y, z] > high):
            continue
        points.append([x, y, z])
        radii.append(10.0)
    points = np.array(points, dtype=np.float32)
    radii = np.array(radii, dtype=np.float32)

    to_keep = vtu_mesh.find_containing_cell(points)
    to_keep += 1
    to_keep = np.argwhere(to_keep)
    points = points[to_keep.reshape(-1)]
    radii = radii[to_keep.reshape(-1)]
    if name.lower() != "aorta":
        points = points[:-15]
        radii = radii[:-15]
    else:
        points = points[7:]
        radii = radii[7:]

    return Branch(
        name=name.lower(),
        coordinates=np.array(points, dtype=np.float32),
        radii=np.array(radii, dtype=np.float32),
    )


class AorticArch:
    def __init__(self) -> None:

        vmr_folder = download_vmr_files(["0094_0001"])
        self._model_folder = os.path.join(vmr_folder, "0094_0001")
        self._make_branches([0, 133, 0])
        self._make_mesh_obj([0, 133, 0])

    def _make_branches(self, rotate_yzx_deg):
        vtu_mesh_path = os.path.join(self._model_folder, "Meshes", "0094_0001.vtu")
        reader = pv.get_reader(vtu_mesh_path)
        vtu_mesh: pv.UnstructuredGrid = reader.read()
        vtu_mesh.scale([SCALING_FACTOR, SCALING_FACTOR, SCALING_FACTOR], inplace=True)

        path_dir = os.path.join(self._model_folder, "Paths")
        pth_files = []
        for file in os.listdir(path_dir):
            if file.endswith(".pth"):
                pth_files.append(file)
        pth_files = sorted(pth_files)
        branches = []
        for file in pth_files:
            file_path = os.path.join(path_dir, file)
            branch = _load_points_from_pth(file_path, vtu_mesh)
            branches.append(branch)

        branches = rotate(branches, rotate_yzx_deg)
        self.branches = tuple(branches)

        aorta = self["aorta"]
        insertion_point, ip_dir = self.calc_insertion(aorta, -6, -8)
        self.insertion = Insertion(insertion_point, ip_dir)
        self.branching_points = calc_branching(self.branches)
        self._mesh_path = None

        branch_highs = [branch.high for branch in branches]
        high = np.max(branch_highs, axis=0)
        branch_lows = [branch.low for branch in branches]
        low = np.min(branch_lows, axis=0)
        self.coordinate_space = gym.spaces.Box(low=low, high=high)

    def _make_mesh_obj(self, rotate_yzx_deg):
        vtp_mesh_path = os.path.join(self._model_folder, "Meshes", "0094_0001.vtp")

        reader = pv.get_reader(vtp_mesh_path)
        vtp_mesh: pv.PolyData = reader.read()
        vtp_mesh.flip_normals()
        vtp_mesh.scale([SCALING_FACTOR, SCALING_FACTOR, SCALING_FACTOR], inplace=True)
        vtp_mesh.rotate_y(rotate_yzx_deg[0], inplace=True)
        vtp_mesh.rotate_z(rotate_yzx_deg[1], inplace=True)
        vtp_mesh.rotate_x(rotate_yzx_deg[2], inplace=True)

        vtp_mesh.decimate(0.99, inplace=True)

        obj_mesh_path = self.get_temp_obj_mesh_path("VMR_0094_0001")
        pv.save_meshio(obj_mesh_path, vtp_mesh)
        self.mesh_path = obj_mesh_path

    @staticmethod
    def get_temp_obj_mesh_path(name_base):
        while True:
            pid = os.getpid()
            nr = int(os.times().elapsed)
            mesh_path = f"{gettempdir()}/{name_base}_{pid}-{nr}.obj"
            if not os.path.exists(mesh_path):
                try:
                    open(mesh_path, "x", encoding="utf-8").close()
                    break
                except IOError:
                    continue
        return mesh_path

    def __getitem__(self, item: Union[int, str]):
        if isinstance(item, int):
            idx = item
        else:
            branch_names = tuple(branch.name for branch in self.branches)
            idx = branch_names.index(item)
        return self.branches[idx]

    def values(self) -> Tuple[Branch]:
        return self.branches

    def keys(self) -> Tuple[str]:
        return tuple(branch.name for branch in self.branches)

    def items(self):
        branch_names = tuple(branch.name for branch in self.branches)
        return zip(branch_names, self.branches)

    @staticmethod
    def calc_insertion(
        branch: Branch, idx_0: int, idx_1: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        point_0 = branch.coordinates[idx_0]
        point_1 = branch.coordinates[idx_1]
        insertion_point = point_0
        insertion_direction = point_1 - point_0
        insertion_direction = insertion_direction / np.linalg.norm(insertion_direction)
        return insertion_point, insertion_direction

    def find_nearest_branch_to_point(self, point: np.ndarray) -> Branch:
        nearest_branch = None
        minDist = np.inf
        for branch in self.branches:
            distances = np.linalg.norm(branch.coordinates - point, axis=1)
            dist = np.min(distances)
            if dist < minDist:
                minDist = dist
                nearest_branch = branch
        return nearest_branch

    def at_tree_end(self, point: np.ndarray):
        branch = self.find_nearest_branch_to_point(point)
        branch_np = branch.coordinates
        distances = np.linalg.norm(branch_np - point, axis=1)
        min_idx = np.argmin(distances)
        sec_min_idx = np.argpartition(distances, 1)[1]
        min_to_sec_min = branch_np[sec_min_idx] - branch_np[min_idx]
        min_to_point = point - branch_np[min_idx]
        dot_prod = np.dot(min_to_sec_min, min_to_point)

        if (min_idx == 0 or min_idx == branch_np.shape[0] - 1) and dot_prod <= 0:
            branch_point = branch.coordinates[min_idx]
            end_is_open = True
            for branching_point in self.branching_points:
                dist = np.linalg.norm(branching_point.coordinates - branch_point)
                if dist < branching_point.radius:
                    end_is_open = False
            return end_is_open
        else:
            return False
