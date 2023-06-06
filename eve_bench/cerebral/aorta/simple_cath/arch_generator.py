import numpy as np
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
