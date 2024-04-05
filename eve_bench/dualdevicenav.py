import os
import json
from pathlib import Path
import numpy as np
from typing import Tuple, List
import eve
from eve.intervention.vesseltree.util.branch import (
    Branch,
    BranchWithRadii,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data/dualdevicenav"


class DualDeviceNav(eve.intervention.MonoPlaneStatic):
    def __init__(
        self,
        stop_device_at_tree_end: bool = True,
        normalize_action: bool = False,
    ) -> None:

        mesh = os.path.join(DATA_DIR, "vessel_architecture_collision.obj")
        visu_mesh = os.path.join(DATA_DIR, "vessel_architecture_visual.obj")

        centerline_folder_path = os.path.join(DATA_DIR, "Centrelines_comb")
        branches = load_branches(centerline_folder_path)

        insertion = [65.0, -5.0, 35.0]

        vessel_tree = eve.intervention.vesseltree.FromMesh(
            mesh,
            insertion,
            [-1.0, 0.0, 1.0],
            branch_list=branches,
            rotation_yzx_deg=[90, -90, 0],
            scaling_xyz=[1.0, 1.0, 1.0],
            rotate_branches=False,
            rotate_ip=False,
            visu_mesh=visu_mesh,
        )

        device1 = eve.intervention.device.JShaped(
            name="mic_guide",
            length=900,
            velocity_limit=(35, 3.14),
            visu_edges_per_mm=0.5,
            tip_outer_diameter=0.36,
            straight_outer_diameter=0.36,
            tip_inner_diameter=0,
            straight_inner_diameter=0.36,
            mass_density_tip=0.000005,
            mass_density_straight=0.000005,
            young_modulus_tip=1e3,
            young_modulus_straight=1e3,
            beams_per_mm_straight=0.6,
        )

        device2 = eve.intervention.device.JShaped(
            name="mic_cath",
            length=900,
            velocity_limit=(35, 3.14),
            visu_edges_per_mm=0.5,
            tip_outer_diameter=0.6,
            straight_outer_diameter=0.7,
            tip_inner_diameter=0.57,
            straight_inner_diameter=0.57,
            color=(1.0, 0.0, 0.0),
            mass_density_tip=0.000005,
            mass_density_straight=0.000005,
            young_modulus_tip=1e3,
            young_modulus_straight=1e3,
            beams_per_mm_straight=0.6,
        )

        simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.001)

        fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
            simulation=simulation,
            vessel_tree=vessel_tree,
            image_frequency=7.5,
            image_rot_zx=[20, 5],
        )

        target = eve.intervention.target.CenterlineRandom(
            vessel_tree=vessel_tree,
            fluoroscopy=fluoroscopy,
            threshold=5,
            branches=[
                "Centerline curve - LCCA.mrk",
                "Centerline curve - LVA.mrk",
                "Centerline curve - RCCA.mrk",
                "Centerline curve - RVA.mrk",
            ],
        )

        super().__init__(
            vessel_tree,
            [device1, device2],
            simulation,
            fluoroscopy,
            target,
            stop_device_at_tree_end,
            normalize_action,
        )


def load_points_from_json(json_file_path: str) -> Tuple[Branch, List[float]]:
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    points = []
    radii = []
    for markup in data["markups"]:
        if markup["type"] == "Curve":
            control_points = markup["controlPoints"]
            for point in control_points:
                position = point["position"]
                x = float(position[0])
                y = float(position[1])
                z = float(position[2])
                points.append((y, -z, -x))  # Append as a tuple instead of a list

            if "measurements" in markup:
                measurements = markup["measurements"]
                for measurement in measurements:
                    if measurement["name"] == "Radius":
                        radii.extend(measurement["controlPointValues"])

    points = np.array(points, dtype=np.float32)
    filename = os.path.splitext(os.path.basename(json_file_path))[0]

    radii = np.array(radii, dtype=np.float32)
    branch = BranchWithRadii(name=filename, coordinates=points, radii=radii)

    return branch


def load_branches(folder_path: str) -> list:
    centerlines = []
    for filename in os.listdir(folder_path):
        if filename.startswith("Centerline curve ") and filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            centerline = load_points_from_json(file_path)
            centerlines.append(centerline)
    return centerlines
