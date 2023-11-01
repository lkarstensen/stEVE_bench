import math
import eve


class VMR_0011_H_AO_H(eve.intervention.MonoPlaneStatic):
    def __init__(
        self,
        stop_device_at_tree_end: bool = True,
        normalize_action: bool = False,
    ) -> None:
        vessel_tree = eve.intervention.vesseltree.VMR(
            "0011_H_AO_H",
            insertion_vessel_name="aorta",
            insertion_point_idx=-1,
            insertion_direction_idx_diff=-2,
            approx_branch_radii=5,
            rotate_yzx_deg=[0, 133, 0],
        )

        device = eve.intervention.device.JShaped(
            name="guidewire",
            velocity_limit=(35, 3.14),
            length=450,
            tip_radius=12.1,
            tip_angle=0.4 * math.pi,
            tip_outer_diameter=0.7,
            tip_inner_diameter=0.0,
            straight_outer_diameter=0.89,
            straight_inner_diameter=0.0,
            poisson_ratio=0.49,
            young_modulus_tip=17e3,
            young_modulus_straight=80e3,
            mass_density_tip=0.000021,
            mass_density_straight=0.000021,
            visu_edges_per_mm=0.5,
            collis_edges_per_mm_tip=2,
            collis_edges_per_mm_straight=0.1,
            beams_per_mm_tip=1.4,
            beams_per_mm_straight=0.5,
            color=(0.0, 0.0, 0.0),
        )

        simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.1)

        fluoroscopy = eve.intervention.fluoroscopy.TrackingOnly(
            simulation=simulation,
            vessel_tree=vessel_tree,
            image_frequency=7.5,
            image_rot_zx=[0, 0],
            image_center=[0, 0, 0],
            field_of_view=None,
        )

        target = eve.intervention.target.CenterlineRandom(
            vessel_tree=vessel_tree,
            fluoroscopy=fluoroscopy,
            threshold=5,
            branches=["btrunk", "carotid", "rt_carotid", "subclavian"],
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
