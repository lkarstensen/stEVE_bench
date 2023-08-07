import math
import eve


class ArchGenerator(eve.intervention.MonoPlaneStatic):
    def __init__(
        self,
        episodes_between_arch_change: int = 3,
        stop_device_at_tree_end: bool = True,
        normalize_action: bool = False,
    ) -> None:
        vessel_tree = eve.intervention.vesseltree.AorticArchRandom(
            episodes_between_change=episodes_between_arch_change
        )
        device = eve.intervention.device.JShaped(
            name="guidewire",
            velocity_limit=(25, 3.14),
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
        device2 = eve.intervention.device.JShaped(
            name="cath",
            velocity_limit=(25, 3.14),
            length=450,
            tip_radius=10.0,
            tip_angle=0.5 * math.pi,
            tip_outer_diameter=1.2,
            tip_inner_diameter=1.0,
            straight_outer_diameter=1.2,
            straight_inner_diameter=1.0,
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
            color=(1.0, 0.0, 0.0),
        )
        simulation = eve.intervention.simulation.SofaBeamAdapter(friction=0.3)

        fluoroscopy = eve.intervention.fluoroscopy.Fluoroscopy(
            simulation=simulation,
            vessel_tree=vessel_tree,
            image_frequency=7.5,
            image_rot_zx=[25, 0],
        )

        target = eve.intervention.target.CenterlineRandom(
            vessel_tree=vessel_tree,
            fluoroscopy=fluoroscopy,
            threshold=5,
            branches=["lcca", "rcca", "lsa", "rsa", "bct", "co"],
        )

        super().__init__(
            vessel_tree,
            [device, device2],
            simulation,
            fluoroscopy,
            target,
            stop_device_at_tree_end,
            normalize_action,
        )

    @property
    def episodes_between_arch_change(self) -> int:
        return self.vessel_tree.episodes_between_change
