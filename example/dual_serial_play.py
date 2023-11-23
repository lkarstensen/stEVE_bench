# pylint: disable=no-member

from time import perf_counter
import pygame

from eve_bench.neurovascular.aorta.guidewire_only.arch_generator import ArchGenerator
from eve.visualisation import SofaPygame
from eve.util.userinput.instrumentaction import SerialTwoDevice
from eve.util.userinput.visumanipulator import VisuManipulator

intervention = ArchGenerator()
visu = SofaPygame(intervention)

#instrumentaction = SerialOneDevice()
instrumentaction = SerialTwoDevice('COM13', 115200)
visumanipulator = VisuManipulator(visu)

n_steps = 0
r_cum = 0.0

intervention.reset()
visu.reset()

while True:
    start = perf_counter()

    pygame.event.get()
    keys_pressed = pygame.key.get_pressed()
    action = instrumentaction.get_action()
    #visumanipulator.step()
    intervention.step(action=action)
    image = visu.render()
    # plt.imshow(image)
    # plt.show()
    n_steps += 1

    if keys_pressed[pygame.K_RETURN]:
        intervention.reset()
        intervention.reset_devices()
        visu.reset()
        n_steps = 0

    print(f"FPS: {1/(perf_counter()-start)}")

    if keys_pressed[pygame.K_ESCAPE]:
        break
intervention.close()
visu.close()
