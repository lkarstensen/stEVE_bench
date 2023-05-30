# pylint: disable=no-member

from time import perf_counter
import pygame
import numpy as np
import eve
import eve.visualisation

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

intervention.save_config(
    "/Users/lennartkarstensen/stacie/eve_bench/bench_candidates/supraaortal_two_device.py"
)

visualisation = eve.visualisation.SofaPygame(intervention)


r_cum = 0.0

intervention.reset()
visualisation.reset()
last_tracking = None
while True:
    start = perf_counter()
    trans = 0.0
    rot = 0.0
    camera_trans = np.array([0.0, 0.0, 0.0])
    camera_rot = np.array([0.0, 0.0, 0.0])
    zoom = 0
    pygame.event.get()
    keys_pressed = pygame.key.get_pressed()

    if keys_pressed[pygame.K_ESCAPE]:
        break
    if keys_pressed[pygame.K_UP]:
        trans += 25
    if keys_pressed[pygame.K_DOWN]:
        trans -= 25
    if keys_pressed[pygame.K_LEFT]:
        rot += 1 * 3.14
    if keys_pressed[pygame.K_RIGHT]:
        rot -= 1 * 3.14
    if keys_pressed[pygame.K_r]:
        lao_rao = 0
        cra_cau = 0
        if keys_pressed[pygame.K_d]:
            lao_rao += 10
        if keys_pressed[pygame.K_a]:
            lao_rao -= 10
        if keys_pressed[pygame.K_w]:
            cra_cau -= 10
        if keys_pressed[pygame.K_s]:
            cra_cau += 10
        visualisation.rotate(lao_rao, cra_cau)
    else:
        if keys_pressed[pygame.K_w]:
            camera_trans += np.array([0.0, 0.0, 200.0])
        if keys_pressed[pygame.K_s]:
            camera_trans -= np.array([0.0, 0.0, 200.0])
        if keys_pressed[pygame.K_a]:
            camera_trans -= np.array([200.0, 0.0, 0.0])
        if keys_pressed[pygame.K_d]:
            camera_trans = np.array([200.0, 0.0, 0.0])
        visualisation.translate(camera_trans)
    if keys_pressed[pygame.K_e]:
        visualisation.zoom(1000)
    if keys_pressed[pygame.K_q]:
        visualisation.zoom(-1000)

    if keys_pressed[pygame.K_v]:
        action = ((0, 0), (trans, rot))

    else:
        action = ((trans, rot), (0, 0))
    intervention.step(action=action)
    visualisation.render()

    if keys_pressed[pygame.K_RETURN]:
        intervention.reset()
        visualisation.reset()
        n_steps = 0

intervention.close()
visualisation.close()
