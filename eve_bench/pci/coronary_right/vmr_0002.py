import math
import eve


class VMR_0002(eve.intervention.MonoPlaneStatic):
    def __init__(
        self,
        stop_device_at_tree_end: bool = True,
        normalize_action: bool = False,
    ) -> None:
        vessel_tree = eve.intervention.vesseltree.VMR(
            "0002_0001",
            insertion_vessel_name="rca",
            insertion_point_idx=2,
            insertion_direction_idx_diff=2,
            approx_branch_radii=10,
            rotate_yzx_deg=[0, 0, 0],
        )
        device = eve.intervention.device.JShaped(
            name="guidewire",
            beams_per_mm_straight=0.5,
            velocity_limit=(25, 3.14),
            tip_outer_diameter=0.3,
            straight_outer_diameter=0.3556,
            tip_radius=5,
            tip_angle=0.2 * math.pi,
        )

        simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.1)

        fluoroscopy = eve.intervention.fluoroscopy.Fluoroscopy(
            simulation=simulation,
            vessel_tree=vessel_tree,
            image_frequency=7.5,
            image_rot_zx=[0, 0],
        )

        target = eve.intervention.target.CenterlineRandom(
            vessel_tree=vessel_tree,
            fluoroscopy=fluoroscopy,
            threshold=2,
            branches=["rca", "rca_b1", "rca_b1_b1", "rca_b2", "rca_b3", "rca_b3_b1"],
        )

        super().__init__(
            vessel_tree,
            [device],
            simulation,
            fluoroscopy,
            target,
            stop_device_at_tree_end,
            normalize_action,
        )
