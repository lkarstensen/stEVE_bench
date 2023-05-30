import numpy as np
import eve

vessel_tree = eve.intervention.vesseltree.VMR(
    "0105_0001",
    insertion_vessel_name="aorta",
    insertion_point_idx=-1,
    insertion_direction_idx_diff=-2,
    approx_branch_radii=10,
    rotate_yzx_deg=[0, 180, 0],
)
device = eve.intervention.device.JShaped(
    name="guidewire", beams_per_mm_straight=0.5, velocity_limit=(25, 3.14)
)
device2 = eve.intervention.device.JShaped(
    name="cath",
    tip_angle=0.5 * np.pi,
    tip_radius=10.0,
    velocity_limit=(25, 3.14),
    tip_outer_diameter=1.2,
    straight_outer_diameter=1.2,
    tip_inner_diameter=1.0,
    straight_inner_diameter=1.0,
    color=(1.0, 0.0, 0.0),
)
simulation = eve.intervention.simulation.Simulation(friction=0.3)

fluoroscopy = eve.intervention.fluoroscopy.Fluoroscopy(
    simulation=simulation,
    vessel_tree=vessel_tree,
    image_frequency=7.5,
    image_rot_zx=[45, 0],
)

target = eve.intervention.target.CenterlineRandom(
    vessel_tree=vessel_tree,
    fluoroscopy=fluoroscopy,
    threshold=5,
    branches=["carotid", "rt_carotid", "subclavian"],
)


intervention = eve.intervention.MonoPlaneStatic(
    vessel_tree=vessel_tree,
    devices=[device, device2],
    simulation=simulation,
    fluoroscopy=fluoroscopy,
    target=target,
)
